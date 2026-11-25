"""Workload W (study doc §3, 6'): the new tokens' K/V of each layer are written to the user's remote slot right
after the layer's K/V exist (model.forward_cached(after_kv=...)); the K/V buffer rows ARE the packed source,
so "pack" costs nothing beyond the index_copy the model already does. Completion is measured as durable:
a write-then-read fence on every QP (an 8-byte read served after the earlier writes on that QP).

Timing stamps (globaltimer, per request): t0 request start, t1 output ready (latency) (main stream), t2 writes durable
(side stream). Writers: NoWrite (compute only), GinWriter (GIN put from a kernel), VerbsWriter (CPU posts the
writes when the GPU flags a layer as ready)."""
import numpy as np
import torch

from .model import Batch


class WriterBase:
    variants = 8
    name = "none"

    def __init__(self, cfg, window, pool_bytes, device="cuda", seed=11):
        self.cfg, self.win, self.pool = cfg, window, pool_bytes
        self.dev = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.scratch = window.numel() - 4096          # fence landing zone: last 4 KiB of the window
        self.side = torch.cuda.Stream(device=self.dev)
        self.evs = [torch.cuda.Event() for _ in range(cfg.layers)]

    def setup(self, b: Batch, blk):
        """blk: bytes of one K (or V) block of the local KV buffer (rows = b.total_k)."""
        cfg = self.cfg
        tok = cfg.hd * 2
        self.b, self.blk, self.tok = b, blk, tok
        slot_len = max(p + s for p, s in zip(b.hist, b.new))
        slot = cfg.layers * 2 * slot_len * tok
        n_users = self.pool // max(slot, 1) - 1          # keep remote offset 0 out of user slots (fence target)
        if n_users < b.B:
            raise torch.cuda.OutOfMemoryError(f"store pool holds {n_users} users < B={b.B}")
        self.tables, self.layer_tables = [], []
        for _ in range(self.variants):
            users = self.rng.choice(n_users, size=b.B, replace=False) + 1
            segs, lp = [], [0]
            for l in range(cfg.layers):
                for u in range(b.B):
                    P, S = b.hist[u], b.new[u]
                    if S == 0:
                        continue
                    for kv in range(2):
                        remote = int(users[u]) * slot + (2 * l + kv) * slot_len * tok + P * tok
                        local = (2 * l + kv) * blk + (b.k_off[u] + P) * tok
                        segs.append([remote, local, S * tok, int(users[u])])
                lp.append(len(segs))
            t = torch.tensor(segs, dtype=torch.int64).reshape(-1, 4)
            self.tables.append(t)
        self.lp = lp
        self.layer_ptr_cpu = torch.tensor(lp, dtype=torch.int32)
        self.cur = 0
        self.bytes = sum(b.new) * cfg.layers * cfg.kv_tok_bytes

    def pre_replay(self, i):
        self.cur = i % self.variants

    def start(self):
        pass

    def after_kv(self, l, k, v):
        pass

    def end(self, stamps):
        """Called after the output's stamp; must leave stamps[2] = durable time and rejoin the side stream."""
        torch.ops.aten.copy_(stamps[2:3], stamps[1:2])

    def readback(self, ext):
        return None


class NoWrite(WriterBase):
    name = "none"


class GinWriter(WriterBase):
    name = "gin"

    def __init__(self, cfg, gin, ext, peer, pool_bytes, ctx0, n_ctx=2, device="cuda"):
        super().__init__(cfg, gin.local(), pool_bytes, device)
        self.g, self.E, self.peer, self.ctx0, self.n_ctx = gin, ext, peer, ctx0, n_ctx

    def setup(self, b, blk):
        super().setup(b, blk)
        self.dev_tables = [t.to(self.dev) for t in self.tables]
        self.segs = self.dev_tables[0].clone()
        self.views = [self.segs[self.lp[l]:self.lp[l + 1]] for l in range(self.cfg.layers)]

    def pre_replay(self, i):
        super().pre_replay(i)
        self.segs.copy_(self.dev_tables[self.cur], non_blocking=True)

    def after_kv(self, l, k, v):
        if self.views[l].numel() == 0:
            return
        main = torch.cuda.current_stream()
        self.evs[l].record(main)
        self.side.wait_event(self.evs[l])
        with torch.cuda.stream(self.side):
            self.g.put_multi(self.views[l], self.peer, self.ctx0, self.n_ctx)

    def end(self, stamps):
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            self.g.fence(self.peer, self.ctx0, self.n_ctx, 0, self.scratch, -1, stamps[2:3])
        main.wait_stream(self.side)

    def readback(self, ext):
        """Fetch the last request's written ranges back into scratch space after the KV buffer; compare."""
        segs = self.tables[self.cur]
        if segs.numel() == 0:
            return 0
        base = int(self.cfg.layers * 2 * self.blk)
        back, off = [], base
        for r, l_, n, u in segs.tolist():
            back.append([r, off, n, u])
            off += n
        if off > self.scratch:
            return None
        t = torch.tensor(back, dtype=torch.int64, device=self.dev)
        lp = torch.tensor([0, len(back)], dtype=torch.int32, device=self.dev)
        tickets = torch.zeros(self.n_ctx, 2, dtype=torch.int64, device=self.dev)
        self.g.post_gets(t, lp, tickets, self.peer, ctx0=self.ctx0, n_ctx=self.n_ctx)
        self.g.wait_layer(tickets, 0, self.ctx0, self.n_ctx)
        torch.cuda.synchronize()
        bad = 0
        for (r, l_, n, u), (_, o, _, _) in zip(segs.tolist(), back):
            bad += int((self.win[l_:l_ + n] != self.win[o:o + n]).sum().item())
        return bad


class VerbsWriter(WriterBase):
    name = "verbs"

    def __init__(self, cfg, verbs, ext, pool_bytes, device="cuda"):
        super().__init__(cfg, verbs.local(), pool_bytes, device)
        self.v, self.E = verbs, ext
        self.seq_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.seq = 0
        self.wflags, self.done = verbs.wflags_ptr(), verbs.done_ptr()

    def setup(self, b, blk):
        super().setup(b, blk)
        self.v.wait_idle()
        # continue the engine's sequence (shared with the read sources): the gate flags and the done flag hold the
        # last finished seq of any earlier writer, so restarting at 0 would let a new cell's first writes pass early
        from .kvsource import VerbsKVSource
        self._eng = VerbsKVSource._engine_seq
        self.seq = self._eng.get(id(self.v), 0)
        self.seq_dev.fill_(self.seq)

    def pre_replay(self, i):
        super().pre_replay(i)
        self.seq += 1
        self._eng[id(self.v)] = self.seq
        self.v.post_writes(self.tables[self.cur], self.layer_ptr_cpu, self.seq, 0, self.scratch)

    def start(self):
        self.E.seq_inc(self.seq_dev)

    def after_kv(self, l, k, v):
        self.E.set_flag_seq(self.wflags, l, self.seq_dev)

    def end(self, stamps):
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            self.E.wait_flags_seq(self.done, 0, self.seq_dev, stamps[2:3])
        main.wait_stream(self.side)

    def readback(self, ext):
        segs = self.tables[self.cur]
        if segs.numel() == 0:
            return 0
        self.v.wait_idle()
        base = int(self.cfg.layers * 2 * self.blk)
        back, off = [], base
        for r, l_, n, u in segs.tolist():
            back.append([r, off, n, u])
            off += n
        if off > self.scratch:
            return None
        self.seq += 1
        self._eng[id(self.v)] = self.seq
        self.v.post_reads(torch.tensor(back, dtype=torch.int64), torch.tensor([0, len(back)], dtype=torch.int32),
                          self.seq)
        self.v.wait_idle()
        torch.cuda.synchronize()
        bad = 0
        for (r, l_, n, u), (_, o, _, _) in zip(segs.tolist(), back):
            bad += int((self.win[l_:l_ + n] != self.win[o:o + n]).sum().item())
        self.seq_dev.fill_(self.seq)     # keep device seq in step with the extra post
        return bad
