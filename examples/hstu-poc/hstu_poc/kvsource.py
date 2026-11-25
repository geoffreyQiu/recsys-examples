"""Where a request's cached history K/V comes from. Every source exposes the same hooks, called by the model
inside one CUDA graph per cell:
    setup(batch)      allocate / plan for this batch shape (outside the graph)
    kv(layer)         (K, V) views [>= total_k, heads, head_dim] of the local GPU buffers
    start()           at request start: begin moving every layer's history (post gets, issue copies, ...)
    before_attn(l)    right before layer l's attention: wait until layer l's history has arrived
    end()             after the last layer (join side streams, ...)
Local GPU buffer layout: [layer][K|V][total_k][heads][head_dim]; user u's history occupies rows
[k_off[u], k_off[u] + L_u) of each block, so one user's history in one layer is two contiguous ranges.
"""
import os

import torch

from .model import Batch, Config


class LocalKV:
    """Local GPU-memory hit: the history is already in GPU memory (moves nothing)."""
    name = "local"

    def __init__(self, cfg: Config, device="cuda", local_buf=None):
        self.cfg, self.dev = cfg, torch.device(device)
        self.local_buf = local_buf      # carve buffers from here when given (a transport's registered window)

    def alloc(self, nbytes):
        if self.local_buf is not None:
            if nbytes > self.local_buf.numel():
                raise torch.cuda.OutOfMemoryError(f"local window {self.local_buf.numel() / 1e9:.1f} GB < {nbytes / 1e9:.1f} GB")
            return self.local_buf[:nbytes]
        return torch.empty(nbytes, dtype=torch.uint8, device=self.dev)

    def setup(self, b: Batch):
        cfg = self.cfg
        self.b = b
        self.blk = b.total_k * cfg.hd * 2                     # bytes of one K (or V) block
        self.buf = self.alloc(cfg.layers * 2 * self.blk)
        self.views = []
        for l in range(cfg.layers):
            k = self.buf[(2 * l) * self.blk:(2 * l + 1) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            v = self.buf[(2 * l + 1) * self.blk:(2 * l + 2) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            self.views.append((k, v))
        self.views[0][0].normal_(0, 0.5)
        for l in range(cfg.layers):        # any finite values will do for timing
            for t in self.views[l]:
                t.copy_(self.views[0][0])

    def kv(self, l):
        return self.views[l]

    def start(self):
        pass

    def before_attn(self, l):
        pass

    def end(self):
        pass

    def bytes_moved(self):
        return 0

    def release(self):
        self.views, self.buf = None, None


class H2DKV(LocalKV):
    """Local host memory -> GPU with copy engines: per user and layer, 2 cudaMemcpyAsync (K, V) on a copy stream
    issued at request start, one event per layer; the compute stream waits on layer l's event."""
    name = "h2d"

    def setup(self, b: Batch):
        super().setup(b)
        cfg = self.cfg
        tok = cfg.hd * 2
        # host store: per user a slot [layer][K|V][L_u] tokens (users at distinct offsets, like distinct users)
        self.slot = [cfg.layers * 2 * L * tok for L in b.hist]
        self.host = torch.empty(sum(self.slot), dtype=torch.uint8, pin_memory=True)
        self.host.view(cfg.dtype)[: min(self.host.numel() // 2, 1 << 24)].normal_(0, 0.5)
        self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.events = [torch.cuda.Event() for _ in range(cfg.layers)]
        self.plan = []                     # per layer: list of (dst view, src view)
        base = 0
        offs = []
        for s in self.slot:
            offs.append(base)
            base += s
        for l in range(cfg.layers):
            ops = []
            for u, L in enumerate(b.hist):
                if L == 0:
                    continue
                for kv in range(2):
                    src = self.host[offs[u] + (2 * l + kv) * L * tok: offs[u] + (2 * l + kv + 1) * L * tok]
                    dst = self.buf[(2 * l + kv) * self.blk + b.k_off[u] * tok:(2 * l + kv) * self.blk + (b.k_off[u] + L) * tok]
                    ops.append((dst, src))
            self.plan.append(ops)

    def start(self):
        main = torch.cuda.current_stream()
        self.copy_stream.wait_stream(main)
        with torch.cuda.stream(self.copy_stream):
            for l, ops in enumerate(self.plan):
                for dst, src in ops:
                    dst.copy_(src, non_blocking=True)
                self.events[l].record(self.copy_stream)

    def before_attn(self, l):
        torch.cuda.current_stream().wait_event(self.events[l])

    def end(self):
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    def bytes_moved(self):
        return sum(self.b.hist) * self.cfg.layers * self.cfg.kv_tok_bytes

    def release(self):
        super().release()
        self.host, self.plan = None, None


class P4:
    """P4 wait variants for the remote sources (study doc §5.4):
      5a     stock FBGEMM kernel; a 1-thread wait kernel before each layer's attention (the default)
      copy   the copied kernel (attn/hstu_hopper) with the same 5'a waits: shows the copy itself costs nothing
      pdl    copied kernel launched with programmatic dependent launch right after the wait kernel, which triggers
             it on entry: the attention prologue overlaps the wait; griddepcontrol.wait before the first K/V load
      fused  no wait kernels: a 1-thread poller on a side stream publishes per-layer ready words (GIN/raw: waits on
             the tickets; CPU verbs: relays the CPU's host flags); the copied kernel's producer warp spins on its
             layer's word before the first K/V load (variant (i))
      pdl_early  early CTA release: a 1-thread kernel triggers the dependent launch and exits at once (its CTA and SM
             are free immediately); the copied kernel, launched with PDL, needs no griddepcontrol.wait (everything it
             reads from earlier kernels completed before the trigger kernel started) and takes the data dependency
             itself, spinning on its layer's ready word like fused
    slow_ns > 0 delays posting (a slow producer) to prove that the waits really wait."""
    p4 = "5a"
    slow_ns = 0
    attn_fn = None

    def p4_init(self, mode="5a", slow_ns=0):
        self.p4, self.slow_ns = mode, slow_ns
        self.ready = torch.zeros(self.cfg.layers, dtype=torch.int32, device=self.dev)
        if not hasattr(self, "seq_dev"):
            self.seq_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.p4_side = torch.cuda.Stream(device=self.dev)
        self.attn_fn = None if mode == "5a" else self._p4_attn
        if mode == "tile":        # per-tile waits: chunks landed per layer, zeroed at each request's start
            self.tsteps = torch.zeros(self.cfg.layers, dtype=torch.int32, device=self.dev)
        if mode == "pdl_early":   # a flag word that is always set: the trigger kernel passes its check and exits
            self._sat = torch.full((self.cfg.layers,), 2**31 - 1, dtype=torch.int32, device=self.dev)
            self._sat_seq = torch.zeros(1, dtype=torch.int32, device=self.dev)
        return self

    @property
    def fused_like(self):
        return self.p4 in ("fused", "pdl_early", "tile")

    def _p4_attn(self, l, q, K, V, b, model):
        from .attn_ext import hstu_fwd
        args = (q, K, V, b.cu_q, b.cu_k, b.max_q, b.max_k, model.cfg.max_len, b.num_targets, model.alpha)
        if self.p4 == "fused":
            return hstu_fwd(*args, kv_ready=self.ready.data_ptr() + 4 * l, kv_seq=self.seq_dev.data_ptr(), mode=1)
        if self.p4 == "tile":
            return hstu_fwd(*args, kv_ready=self.tsteps.data_ptr() + 4 * l, kv_seq=self.seq_dev.data_ptr(),
                            mode=3 | (self.tile_rows << 8))
        if self.p4 == "pdl_early":
            self.E.wait_flags_seq(self._sat.data_ptr(), 0, self._sat_seq)    # trigger dependents, exit at once
            return hstu_fwd(*args, kv_ready=self.ready.data_ptr() + 4 * l, kv_seq=self.seq_dev.data_ptr(), mode=1,
                            pdl=True)
        self.before_attn(l)
        if self.p4 == "pdl":
            return hstu_fwd(*args, mode=2, pdl=True)
        return hstu_fwd(*args)

    def p4_fork(self, launch):
        main = torch.cuda.current_stream()
        self.p4_side.wait_stream(main)
        with torch.cuda.stream(self.p4_side):
            launch()

    def p4_join(self):
        if self.fused_like or getattr(self, "side_post", False):
            torch.cuda.current_stream().wait_stream(self.p4_side)


class RemoteKV(LocalKV, P4):
    """History in a remote store's memory (one slot per user: [layer][K|V][slot_len][heads][head_dim]).
    Subclasses move it with a transport; this class builds the transfer plan. Users are distinct per request
    and rotate over `variants` user sets between requests (production-like scattered reads)."""
    variants = 8

    base = 0               # byte offset of this source's buffers inside the transport window (P5 regions)
    post_outside = False   # P5: the transfer is posted by post() outside the compute graph (fetch at admission)
    # GDAKI latency options (no effect for CPU verbs):
    zc = False             # zero-copy request descriptor: the post kernel reads the request's table index from pinned
                           # host memory written by the CPU (no per-request device copy of the segment table)
    side_post = False      # the post kernel runs on a side branch of the graph, so the compute starts at once; the
                           # main stream joins it before layer 0's wait (5a/pdl) or only at the end (fused-like)

    def __init__(self, cfg, local_buf, pool_bytes, device="cuda", seed=7):
        super().__init__(cfg, device, local_buf)
        self.pool_bytes = pool_bytes
        self.rng = __import__("numpy").random.default_rng(seed)

    def alloc(self, nbytes):
        if self.base + nbytes > self.local_buf.numel():
            raise torch.cuda.OutOfMemoryError(f"local window {self.local_buf.numel() / 1e9:.1f} GB < {(self.base + nbytes) / 1e9:.1f} GB")
        return self.local_buf[self.base:self.base + nbytes]

    def setup(self, b: Batch):
        cfg = self.cfg
        self.b = b
        self.blk = b.total_k * cfg.hd * 2
        self.buf = self.alloc(cfg.layers * 2 * self.blk)
        self.views = []
        for l in range(cfg.layers):
            k = self.buf[(2 * l) * self.blk:(2 * l + 1) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            v = self.buf[(2 * l + 1) * self.blk:(2 * l + 2) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            self.views.append((k, v))
        tok = cfg.hd * 2
        slot_len = max(b.hist)
        slot = cfg.layers * 2 * slot_len * tok
        n_users = self.pool_bytes // max(slot, 1)
        if n_users < b.B:
            raise torch.cuda.OutOfMemoryError(f"store pool holds {n_users} users of {slot / 2**20:.0f} MiB < B={b.B}")
        self.tables, self.checks = [], []
        for _ in range(self.variants):
            users = self.rng.choice(n_users, size=b.B, replace=False)
            segs, rows, lp = [], [], [0]
            for l in range(cfg.layers):
                for u, L in enumerate(b.hist):
                    if L == 0:
                        continue
                    for kv in range(2):
                        remote = int(users[u]) * slot + (2 * l + kv) * slot_len * tok
                        local = self.base + (2 * l + kv) * self.blk + b.k_off[u] * tok
                        segs.append([remote, local, L * tok, int(users[u])])
                        rows.append([local, L * tok, 0, remote // 8])     # store pattern is by word offset
                lp.append(len(segs))
            self.tables.append(torch.tensor(segs, dtype=torch.int64))
            self.checks.append(torch.tensor(rows, dtype=torch.int64))
        self.layer_ptr_cpu = torch.tensor(lp, dtype=torch.int32)
        self.cur = 0
        # P5 sampled data check: per variant, the last layer's rows (it lands last) of up to 8 of its segments,
        # first 256 KiB of each
        nl = lp[-1] - lp[-2]
        self._samples = []
        for rows in self.checks:
            last = rows[len(rows) - nl:]
            idx = torch.linspace(0, len(last) - 1, steps=min(8, len(last))).round().long()
            t = last[idx].clone()
            t[:, 1] = torch.clamp(t[:, 1], max=256 << 10)
            self._samples.append(t.to(self.dev))

    def sample_rows(self):
        """Rows for a sampled data check of the current request (see setup)."""
        return self._samples[self.cur]

    def _zc_setup(self):
        self.nseg = int(self.tables[0].shape[0])
        self.segs_all = torch.cat([t.to(self.dev) for t in self.tables]).contiguous()
        self.var_host = torch.zeros(1, dtype=torch.int32).pin_memory()
        self.var_np = self.var_host.numpy()

    def _select(self, i):
        """Per request (outside the graph): pick the user set; zero-copy = one store into pinned memory."""
        self.cur = i % self.variants
        if self.zc:
            self.var_np[0] = self.cur
        else:
            self.segs.copy_(self.dev_tables[self.cur], non_blocking=True)

    def _gdaki_start(self):
        """start() of the GDAKI sources (raw, GIN): post every get, plus the fused-like poller."""
        if self.slow_ns and not self.side_post:
            self.E.spin(self.slow_ns)
        if self.fused_like:
            self.E.seq_inc(self.seq_dev)
        if self.post_outside:
            if self.fused_like:
                self.p4_fork(self._poll)
            return
        if self.side_post:
            def side():
                if self.slow_ns:
                    self.E.spin(self.slow_ns)   # a slow producer delays the post only; the compute runs ahead
                self.post()
                if self.fused_like:
                    self._poll()
                else:
                    self._post_ev.record()
            self.p4_fork(side)
        else:
            self.post()
            if self.fused_like:
                self.p4_fork(self._poll)

    def _side_join(self, l):
        if self.side_post and l == 0 and not self.fused_like and not self.post_outside:
            torch.cuda.current_stream().wait_event(self._post_ev)   # tickets exist before the first wait reads them

    def bytes_moved(self):
        return sum(self.b.hist) * self.cfg.layers * self.cfg.kv_tok_bytes

    def check(self, ext, seed):
        """Data check of the last request's history rows against the store pattern (mismatched words)."""
        rows = self.checks[self.cur].to(self.dev)
        bad = torch.zeros(1, dtype=torch.int64, device=self.dev)
        ext.check_pattern_gpu(self.local_buf, rows, seed, bad)
        return int(bad.item())

    def release(self):
        self.views, self.buf, self.tables, self.checks = None, None, None, None


class GinKVSource(RemoteKV):
    """GIN (GDAKI or Proxy, by NCCL_GIN_TYPE), 5'a: a post kernel at request start issues every get and stores
    per-layer tickets; a 1-thread wait kernel before each layer's attention."""
    name = "gin"

    adapt_ctx = False   # use only as many contexts as a layer has segments (no flush/wait on empty contexts)

    def __init__(self, cfg, gin, ext, peer, pool_bytes, n_ctx=2, device="cuda"):
        super().__init__(cfg, gin.local(), pool_bytes, device)
        self.g, self.E, self.peer, self.n_ctx = gin, ext, peer, n_ctx

    def setup(self, b):
        super().setup(b)
        self.segs = self.tables[0].to(self.dev)
        self.dev_tables = [t.to(self.dev) for t in self.tables]
        self.lp = self.layer_ptr_cpu.to(self.dev)
        self.tickets = torch.zeros(self.cfg.layers * self.n_ctx, 2, dtype=torch.int64, device=self.dev)
        per_layer = int(self.layer_ptr_cpu[1] - self.layer_ptr_cpu[0])
        self.nc = min(self.n_ctx, max(per_layer, 1)) if self.adapt_ctx else self.n_ctx
        self._post_ev = torch.cuda.Event()
        if self.zc:
            self._zc_setup()

    def pre_replay(self, i):
        self._select(i)

    def start(self):
        self._gdaki_start()

    def _poll(self):
        self.g.poll(self.tickets, self.cfg.layers, 0, self.nc, self.seq_dev, self.ready)

    def post(self):
        if self.zc:
            self.g.post_gets(self.segs_all, self.lp, self.tickets, self.peer, ctx0=0, n_ctx=self.nc,
                             var=self.var_host, var_stride=self.nseg)
        else:
            self.g.post_gets(self.segs, self.lp, self.tickets, self.peer, ctx0=0, n_ctx=self.nc)

    def before_attn(self, l):
        self._side_join(l)
        self.g.wait_layer(self.tickets, l, 0, self.nc)

    def end(self):
        self.p4_join()


class VerbsKVSource(RemoteKV):
    """CPU verbs, 5' for CPU verbs: the CPU posts every read before the request's graph launches; its worker
    thread polls the CQ and stores seq into flags[l] per completed layer; the graph's first kernel increments
    the device-side seq and each layer's attention waits for flags[l] >= seq (1-thread spin kernel)."""
    name = "verbs"

    def __init__(self, cfg, verbs, ext, pool_bytes, device="cuda"):
        super().__init__(cfg, verbs.local(), pool_bytes, device)
        self.v, self.E = verbs, ext
        self.seq_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.seq = 0
        self.flags = verbs.flags_ptr()

    _engine_seq = {}   # id(engine) -> last seq posted: the worker's flags hold the last finished seq of ANY source

    tile_rows = 0
    max_tile_wrs = 2048   # per request: keeps the reads within the send queues and the completion queue

    def setup(self, b):
        super().setup(b)
        self.v.wait_idle()
        if self.p4 == "tile":   # chunk rows: >= 128, sized so a request posts at most max_tile_wrs reads
            L = max(b.hist)
            rows = max(int(os.environ.get("HPOC_TILE_ROWS", 128)), -(-self.cfg.layers * 2 * b.B * L // self.max_tile_wrs))
            self.tile_rows = -(-rows // 64) * 64
            self.tile_bytes = self.tile_rows * self.cfg.hd * 2
        # continue the engine's sequence, so this source's waits (flags[l] >= seq) cannot pass on an earlier
        # source's larger flags (that let a new cell's first requests run before their data had landed)
        self.seq = VerbsKVSource._engine_seq.get(id(self.v), 0)
        self.seq_dev.fill_(self.seq)

    def pre_replay(self, i):
        self.cur = i % self.variants
        self.seq += 1
        VerbsKVSource._engine_seq[id(self.v)] = self.seq
        if not self.post_after_launch:
            self._post_now()

    post_after_launch = False   # slow-producer check: the caller launches the graph, then calls post_now()

    def post_now(self):
        self._post_now()

    def _post_now(self):
        if self.p4 == "tile":
            self.v.post_reads_tiled(self.tables[self.cur], self.layer_ptr_cpu, self.seq, self.tile_bytes)
        else:
            self.v.post_reads(self.tables[self.cur], self.layer_ptr_cpu, self.seq)

    def start(self):
        self.E.seq_inc(self.seq_dev)
        if self.p4 == "tile":
            self.tsteps.zero_()
            self.p4_fork(lambda: self.E.relay_tiles(self.v.progress_ptr(), self.flags, self.cfg.layers, self.seq_dev,
                                                    self.tsteps))
        elif self.fused_like:
            self.p4_fork(lambda: self.E.relay_flags(self.flags, self.cfg.layers, self.seq_dev, self.ready))

    def before_attn(self, l):
        self.E.wait_flags_seq(self.flags, l, self.seq_dev)

    def end(self):
        self.p4_join()

    def release(self):
        self.v.wait_idle()
        super().release()


class RawGdakiSource(RemoteKV):
    """Raw GDAKI (DOCA GPUNetIO, no NCCL), 5'a like GIN: a post kernel issues every get (spread over the QPs) and
    keeps one ticket per (layer, QP); a 1-thread wait kernel per layer. The store is a GPU-less verbs pool."""
    name = "raw"

    def __init__(self, cfg, raw, ext, pool_bytes, device="cuda"):
        super().__init__(cfg, raw.local(), pool_bytes, device)
        self.r, self.E = raw, ext

    def setup(self, b):
        super().setup(b)
        self.dev_tables = [t.to(self.dev) for t in self.tables]
        self.segs = self.dev_tables[0].clone()
        self.lp = self.layer_ptr_cpu.to(self.dev)
        self.tickets = torch.zeros(self.cfg.layers * self.r.nqp(), dtype=torch.int64, device=self.dev)
        self._post_ev = torch.cuda.Event()
        if self.zc:
            self._zc_setup()

    def pre_replay(self, i):
        self._select(i)

    def start(self):
        self._gdaki_start()

    def _poll(self):
        self.r.poll(self.tickets, self.cfg.layers, self.seq_dev, self.ready)

    def post(self):
        if self.zc:
            self.r.post_gets(self.segs_all, self.lp, self.tickets, var=self.var_host, var_stride=self.nseg)
        else:
            self.r.post_gets(self.segs, self.lp, self.tickets)

    def before_attn(self, l):
        self._side_join(l)
        self.r.wait_layer(self.tickets, l)

    def end(self):
        self.p4_join()
