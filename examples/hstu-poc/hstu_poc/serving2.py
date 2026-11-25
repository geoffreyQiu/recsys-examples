"""Overlap-optimized serving: each strategy on its own, scheduled for the most throughput within a latency SLO.

Remote fetch (FetchServer):
- Every request gets a KV slot in a ring inside the transport window, and its transfer is posted the moment it gets
  a slot (on arrival when slots are free), so the NIC always works on the queue in arrival order while earlier
  requests compute. One transfer per request, every layer posted at once and split in chunks over all queues.
- Compute batches are formed from the oldest posted requests whose data has landed, or will have landed by the time
  the GPU gets to them (a FIFO model of the NIC at its nominal rate); with an idle GPU the oldest request starts at
  once, and its per-layer waits overlap its own transfer. Up to `depth` batches are queued on the GPU, the next one
  formed when the GPU is about to free up, so the GPU never idles between batches.
- One CUDA graph per batch size B, replayed for any group of consecutive slots: K/V are the whole ring, the
  attention's cu_seqlens_k and the candidates' row indices are device tensors rewritten before each replay, and the
  per-layer waits read the batch's last request's tickets (GDAKI) or sequence number (CPU verbs). Completion is in
  order per queue, so the last request's layer-l completion implies every earlier request's.
Recompute (RecomputeServer): dynamic batching to the largest power-of-two batch whose compute fits half the SLO,
  `depth` batches queued, the next one formed when the GPU is about to free up.
Both: graphs replayed once before serving; latency = arrival -> output ready; sampled data checks for fetch.
"""
import time
from collections import deque

import numpy as np
import torch

from .kvsource import VerbsKVSource
from .model import Batch
from .serving import thread_cpu


def pow2_floor(n):
    return 1 << (int(n).bit_length() - 1)


class Ring:
    """Per-request KV slots: per layer a K block and a V block of n_slots x R rows (R = L + C: the history rows, then
    the candidates' rows written by the forward). Slot s = rows [s R, (s+1) R) of every block."""

    def __init__(self, cfg, window, L, C, n_slots):
        self.cfg, self.L, self.C, self.R, self.n = cfg, L, C, L + C, n_slots
        self.row = cfg.hd * 2                                  # bytes of one token's K (or V), all heads
        self.blk = n_slots * self.R * self.row
        need = cfg.layers * 2 * self.blk
        if need > window.numel():
            raise torch.cuda.OutOfMemoryError(f"ring of {n_slots} slots needs {need / 2**30:.1f} GiB")
        self.window = window
        self.views = []
        for l in range(cfg.layers):
            k = window[(2 * l) * self.blk:(2 * l + 1) * self.blk].view(cfg.dtype).view(n_slots * self.R, cfg.heads, cfg.head_dim)
            v = window[(2 * l + 1) * self.blk:(2 * l + 2) * self.blk].view(cfg.dtype).view(n_slots * self.R, cfg.heads, cfg.head_dim)
            self.views.append((k, v))

    def kv(self, l):
        return self.views[l]

    def plan(self, s, user, chunks):
        """Segments [remote, local, bytes, user] of slot s's transfer (user's history in the store), layer by layer;
        each K and V block in `chunks` pieces so every layer spreads over all queues. Returns (segs, layer_ptr)."""
        L, row = self.L, self.row
        store_slot = self.cfg.layers * 2 * L * row
        rows_per = -(-L // chunks)
        segs, lp = [], [0]
        for l in range(self.cfg.layers):
            for kv in range(2):
                r0 = user * store_slot + (2 * l + kv) * L * row
                l0 = (2 * l + kv) * self.blk + s * self.R * row
                for c in range(chunks):
                    a, b = c * rows_per, min(L, (c + 1) * rows_per)
                    if b > a:
                        segs.append([r0 + a * row, l0 + a * row, (b - a) * row, user])
            lp.append(len(segs))
        return segs, lp


class _Common:
    def _account(self, rec, sizes, n, c0, wall, warmup_s, slo_ms, gpu_ms, extra):
        c1 = thread_cpu()
        import threading
        me = threading.get_native_id()
        self.cpu = dict(wall_s=wall, main_s=c1.get(me, 0) - c0.get(me, 0), requests=n,
                        other_s=sum(v - c0.get(t, 0) for t, v in c1.items() if t != me))
        rec = np.array(rec)
        kept = rec[rec[:, 0] >= warmup_s]
        lat = kept[:, 1]
        span = float((kept[:, 0] + kept[:, 1]).max() - warmup_s)
        ok = (lat <= slo_ms / 1e3) & (kept[:, 2] == 0)
        res = dict(lat=lat, achieved=len(kept) / max(span, 1e-9), mean_batch=float(np.mean(sizes)), batches=len(sizes),
                   slo_ms=slo_ms, success=float(ok.mean()), goodput=float(ok.sum()) / max(span, 1e-9),
                   gpu_share=gpu_ms / 1e3 / wall, err_requests=int(kept[:, 2].sum()))
        res.update(extra)
        return res


class FetchServer(_Common):
    def __init__(self, model, cfg, L, C, kind, tr, E, window, pool_bytes, peer=None, n_ctx=4, max_batch=32, depth=2,
                 chunks=2, n_slots=None, nic_gbs=49.4, check_every=8, seed=99, lead_ms=0.5, fetch_stream=None,
                 p4="5a"):
        """kind: 'raw' (raw GDAKI), 'gin' (GIN-GDAKI or GIN-Proxy by NCCL_GIN_TYPE), 'verbs' (CPU verbs).
        p4: '5a' = a 1-thread wait kernel before each layer's attention; 'fused' = no per-layer kernels: a 1-thread
        poller on a side branch of each batch graph publishes per-layer ready words (GDAKI: from the batch's newest
        tickets; CPU verbs: relayed from the host flags) and the attention kernel spins on its layer's word."""
        self.model, self.cfg, self.L, self.C, self.kind, self.tr, self.E = model, cfg, L, C, kind, tr, E
        self.peer, self.n_ctx, self.depth, self.lead = peer, n_ctx, depth, lead_ms / 1e3
        self.check_every, self.seed = check_every, seed
        slot_bytes = cfg.layers * 2 * (L + C) * cfg.hd * 2
        cap = window.numel() // slot_bytes
        self.n_slots = int(min(n_slots or 256, cap))
        if self.n_slots < 2:
            raise torch.cuda.OutOfMemoryError("window holds fewer than 2 slots")
        self.ring = Ring(cfg, window, L, C, self.n_slots)
        self.bytes_per_req = cfg.layers * 2 * L * cfg.hd * 2
        self.t_xfer = self.bytes_per_req / (nic_gbs * 1e9)
        store_users = pool_bytes // (cfg.layers * 2 * L * cfg.hd * 2)
        dev = torch.device("cuda")
        self.lp_cpu, self.segs_cpu, self.segs_dev, self.check_rows = None, [], [], []
        for s in range(self.n_slots):
            user = (s * 7919) % store_users
            segs, lp = self.ring.plan(s, user, chunks)
            t = torch.tensor(segs, dtype=torch.int64)
            self.segs_cpu.append(t)
            if kind != "verbs":
                self.segs_dev.append(t.to(dev))
            seg = t[lp[-2]:lp[-1]]                             # the last layer lands last
            rows = torch.stack([seg[:, 1], torch.clamp(seg[:, 2], max=256 << 10), torch.zeros_like(seg[:, 0]),
                                seg[:, 0] // 8], 1)               # [local, bytes, user 0, store word offset]
            self.check_rows.append(rows.to(dev))
        self.lp_cpu = torch.tensor(lp, dtype=torch.int32)
        self.lp_dev = self.lp_cpu.to(dev)
        if kind == "raw":
            nq = tr.nqp()
            self.tickets = [torch.zeros(cfg.layers * nq, dtype=torch.int64, device=dev) for _ in range(self.n_slots)]
            self.bt = torch.zeros(cfg.layers * nq, dtype=torch.int64, device=dev)
            self._wait = lambda l: tr.wait_layer(self.bt, l)  # noqa: E731
        elif kind == "gin":
            self.tickets = [torch.zeros(cfg.layers * n_ctx, 2, dtype=torch.int64, device=dev) for _ in range(self.n_slots)]
            self.bt = torch.zeros(cfg.layers * n_ctx, 2, dtype=torch.int64, device=dev)
            self._wait = lambda l: tr.wait_layer(self.bt, l, 0, n_ctx)  # noqa: E731
        else:
            self.seqs = [0] * self.n_slots
            self.seq_dev = torch.zeros(1, dtype=torch.int32, device=dev)
            self.flags = tr.flags_ptr()
            self._wait = lambda l: E.wait_flags_seq(self.flags, l, self.seq_dev)  # noqa: E731
        self.p4 = p4
        if p4 == "fused":
            self.ready = torch.zeros(cfg.layers, dtype=torch.int32, device=dev)
            self.side = torch.cuda.Stream()
            if kind == "verbs":
                self.kseq = self.seq_dev                       # the batch's newest request's sequence number
                self._poll = lambda: E.relay_flags(self.flags, cfg.layers, self.seq_dev, self.ready)  # noqa: E731
            else:
                self.kseq = torch.zeros(1, dtype=torch.int32, device=dev)   # incremented by every batch graph
                if kind == "raw":
                    self._poll = lambda: tr.poll(self.bt, cfg.layers, self.kseq, self.ready)  # noqa: E731
                else:
                    self._poll = lambda: tr.poll(self.bt, cfg.layers, 0, n_ctx, self.kseq, self.ready)  # noqa: E731
        elif p4 != "5a":
            raise ValueError(f"serving p4 variant {p4}: 5a or fused")
        self.posted = [torch.cuda.Event() for _ in range(self.n_slots)]
        self.fs = fetch_stream or torch.cuda.Stream()      # GDAKI post kernels (P6: a small green context)
        self.check_stream = torch.cuda.Stream()
        self._bad_dev = torch.zeros(8, dtype=torch.int64, device=dev)
        self._bad_host = torch.zeros(8, dtype=torch.int64).pin_memory()
        self._slot = 0
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = [b for b in (1, 2, 4, 8, 16, 32, 64) if b <= min(max_batch, self.n_slots)]
        self.ex = {B: _FetchExec(self, B) for B in self.buckets}
        # compute time per batch size with the data already in place (graph replayed after its transfer landed)
        self.compute_ms = {}
        for B in self.buckets:
            ex = self.ex[B]
            for _ in range(2):
                self._post_range(0, B)
                ex.launch(0, B - 1)
                torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(3):
                e0.record()
                ex.graph.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            self.compute_ms[B] = float(np.median(ts))
        self.slack = self.compute_ms[1] / 1e3
        nic_ceiling = nic_gbs * 1e9 / self.bytes_per_req
        gpu_ceiling = max(B / (self.compute_ms[B] / 1e3) for B in self.buckets)
        self.ceiling = min(nic_ceiling, gpu_ceiling)
        self.nic_ceiling, self.gpu_ceiling = nic_ceiling, gpu_ceiling

    def _attn(self, l, q, K, V, b, model):
        from .attn_ext import hstu_fwd
        return hstu_fwd(q, K, V, b.cu_q, b.cu_k, b.max_q, b.max_k, model.cfg.max_len, b.num_targets, model.alpha,
                        kv_ready=self.ready.data_ptr() + 4 * l, kv_seq=self.kseq.data_ptr(), mode=1)

    def _post(self, s):
        if self.kind == "verbs":
            key = id(self.tr)
            seq = VerbsKVSource._engine_seq.get(key, 0) + 1
            VerbsKVSource._engine_seq[key] = seq
            self.tr.post_reads(self.segs_cpu[s], self.lp_cpu, seq)
            self.seqs[s] = seq
            return
        with torch.cuda.stream(self.fs):
            if self.kind == "raw":
                self.tr.post_gets(self.segs_dev[s], self.lp_dev, self.tickets[s])
            else:
                self.tr.post_gets(self.segs_dev[s], self.lp_dev, self.tickets[s], self.peer, ctx0=0, n_ctx=self.n_ctx)
            self.posted[s].record(self.fs)

    def _post_range(self, s0, B):
        for s in range(s0, s0 + B):
            self._post(s)

    def _arm(self, last):
        """Point the per-layer waits at slot `last` (the batch's newest request), on the compute stream."""
        main = torch.cuda.current_stream()
        if self.kind == "verbs":
            self.seq_dev.fill_(self.seqs[last])
        else:
            main.wait_event(self.posted[last])
            self.bt.copy_(self.tickets[last])

    def _check(self, s0, B, after):
        """Sampled data check of the batch's newest request (its data lands last), on a side stream."""
        k = self._slot % self._bad_dev.numel()
        self._slot += 1
        rows = self.check_rows[s0 + B - 1]
        with torch.cuda.stream(self.check_stream):
            self.check_stream.wait_event(after)
            self._bad_dev[k:k + 1].zero_()
            self.E.check_pattern_gpu(self.ring.window, rows, self.seed, self._bad_dev[k:k + 1])
            self._bad_host[k:k + 1].copy_(self._bad_dev[k:k + 1], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.check_stream)
        return ev, k

    def run(self, qps, duration_s, warmup_s=1.0, seed=0, slo_ms=100.0, cotenant=None):
        """cotenant: a Cotenant kept busy (two units queued) during the run; its completed units are counted."""
        rng = np.random.default_rng(seed)
        n = int(qps * (duration_s + warmup_s))
        arrivals = np.cumsum(rng.exponential(1.0 / qps, size=n))
        cot, cot_done = deque(), 0
        waiting, posted, inflight, finished = deque(), deque(), deque(), deque()
        head, used, nic_free = 0, 0, 0.0
        rec, sizes = [], []
        gpu_ms = 0.0
        n_checks = bad_words = 0
        nb = 0
        i = 0
        c0 = thread_cpu()
        t0 = time.perf_counter()
        while i < n or waiting or posted or inflight or finished:
            now = time.perf_counter() - t0
            if cotenant is not None:
                while cot and cot[0].query():
                    cot.popleft()
                    cot_done += 1
                while len(cot) < 2:
                    cot.append(cotenant.launch())
            while i < n and arrivals[i] <= now:
                waiting.append(arrivals[i])
                i += 1
            while inflight and inflight[0]["done"].query():           # batches finish in order
                b = inflight.popleft()
                t = time.perf_counter() - t0
                b["t"] = t
                if self.check_every and b["k"] % self.check_every == 0:
                    b["chk"] = self._check(b["s0"], b["B"], b["done"])
                finished.append(b)
            while finished and (finished[0].get("chk") is None or finished[0]["chk"][0].query()):
                b = finished.popleft()
                err = 0
                if b.get("chk") is not None:
                    bad = int(self._bad_host[b["chk"][1]])
                    n_checks += 1
                    bad_words += bad
                    err = int(bad > 0)
                rec.extend((a, b["t"] - a, err) for a in b["arr"])
                used -= b["B"]                                         # slots free in ring order
            while waiting and used < self.n_slots:                     # post transfers in arrival order
                s = head
                head = (head + 1) % self.n_slots
                used += 1
                a = waiting.popleft()
                self._post(s)
                t = time.perf_counter() - t0
                nic_free = max(nic_free, t) + self.t_xfer
                posted.append((s, a, nic_free))
            while posted and len(inflight) < self.depth:              # form compute batches
                t_start = now if not inflight else max(now, inflight[-1]["fin"])
                if inflight and t_start - now > self.lead:
                    break
                k = 0
                for s, a, land in posted:
                    if land > t_start + self.slack:
                        break
                    k += 1
                if k == 0:
                    if inflight:
                        break
                    k = 1                                              # idle GPU: start now, waits overlap the transfer
                s0 = posted[0][0]
                B = pow2_floor(min(k, self.n_slots - s0, self.buckets[-1]))
                batch = [posted.popleft() for _ in range(B)]
                ex = self.ex[B]
                ex.launch(s0, batch[-1][0])
                land_last = batch[-1][2]
                c = self.compute_ms[B] / 1e3
                fin = max(t_start + c, land_last + c / self.cfg.layers)
                inflight.append(dict(done=ex.done, s0=s0, B=B, arr=[a for _, a, _ in batch], fin=fin, k=nb))
                ex.done = torch.cuda.Event()                           # each batch keeps its own completion event
                nb += 1
                sizes.append(B)
                gpu_ms += self.compute_ms[B]
        wall = time.perf_counter() - t0
        if cotenant is not None:
            torch.cuda.synchronize()
        return self._account(rec, sizes, n, c0, wall, warmup_s, slo_ms, gpu_ms,
                             dict(checks=n_checks, bad_words=bad_words, n_slots=self.n_slots,
                                  cotenant_rate=cot_done / wall if cotenant is not None else None))

    def close(self):
        torch.cuda.synchronize()
        if self.kind == "verbs":
            self.tr.wait_idle()
        for ex in self.ex.values():
            ex.close()
        self.ex = {}
        torch.cuda.synchronize()


class _FetchExec:
    """One CUDA graph for batch size B over the ring; set_slots() re-points it at consecutive slots s0 .. s0+B-1."""

    def __init__(self, srv, B):
        cfg, model, L, C = srv.cfg, srv.model, srv.L, srv.C
        self.srv, self.B, self.R = srv, B, srv.ring.R
        dev = torch.device("cuda")
        b = Batch([L] * B, [C] * B, device=dev)
        self.b = b
        self.cu_k0, self.q_dst0 = b.cu_k.clone(), b.q_dst.clone()
        items = torch.randint(0, cfg.n_items, (b.total_q,), device=dev)

        def fwd():
            if srv.p4 == "fused":
                if srv.kind != "verbs":
                    srv.E.seq_inc(srv.kseq)
                srv.side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(srv.side):
                    srv._poll()
                x = model.embed(items, None, b.pos_q)
                y = model.forward_cached(x, b, srv.ring.kv, attn_fn=srv._attn)
                torch.cuda.current_stream().wait_stream(srv.side)
                return y
            x = model.embed(items, None, b.pos_q)
            return model.forward_cached(x, b, srv.ring.kv, before_attn=srv._wait)
        self.items = items
        self._keep = (fwd, items)
        self.done = torch.cuda.Event()
        srv._post_range(0, B)
        srv._arm(B - 1)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fwd()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=srv.pool):
            self.out = fwd()                    # the graph's output buffer (valid after each replay)
        torch.cuda.synchronize()

    def set_slots(self, s0):
        torch.add(self.cu_k0, s0 * self.R, out=self.b.cu_k)
        torch.add(self.q_dst0, s0 * self.R, out=self.b.q_dst)

    def launch(self, s0, last):
        self.set_slots(s0)
        self.srv._arm(last)
        self.graph.replay()
        self.done.record()

    def close(self):
        self.graph.reset()
        self._keep, self.out = None, None


class Cotenant:
    """A co-located GPU job for P6: back-to-back recompute batches (config cfg, B users of history L) replayed on
    their own stream (optionally a green-context stream that leaves some SMs free)."""

    def __init__(self, model, cfg, L, B, C=256, stream=None):
        srv = type("S", (), {})()
        srv.cfg, srv.model, srv.L, srv.C = cfg, model, L, C
        srv.pool = torch.cuda.graph_pool_handle()
        self.stream = stream or torch.cuda.Stream()
        # capture on the job's own stream: a graph captured in the primary context and replayed on a green-context
        # stream still uses every SM; captured on the green stream, its kernels stay in that partition
        self.ex = _RecomputeExec(srv, B, stream=self.stream)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            self.ex.graph.replay()
            e0.record(self.stream)
            self.ex.graph.replay()
            e1.record(self.stream)
        torch.cuda.synchronize()
        self.unit_ms = e0.elapsed_time(e1)

    def launch(self):
        with torch.cuda.stream(self.stream):
            self.ex.graph.replay()
            ev = torch.cuda.Event()
            ev.record(self.stream)
        return ev

    def rate_alone(self, seconds=3.0):
        torch.cuda.synchronize()
        t0, k, q = time.perf_counter(), 0, deque()
        while time.perf_counter() - t0 < seconds:
            while q and q[0].query():
                q.popleft()
                k += 1
            while len(q) < 2:
                q.append(self.launch())
        torch.cuda.synchronize()
        return k / (time.perf_counter() - t0)

    def close(self):
        self.ex.close()


class RecomputeServer(_Common):
    def __init__(self, model, cfg, L, C, max_batch=32, depth=2, slo_ms=100.0, lead_ms=0.5):
        self.model, self.cfg, self.L, self.C, self.depth, self.lead = model, cfg, L, C, depth, lead_ms / 1e3
        self.pool = torch.cuda.graph_pool_handle()
        self.ex, self.compute_ms = {}, {}
        for B in (1, 2, 4, 8, 16, 32, 64):
            if B > max_batch:
                break
            try:
                ex = _RecomputeExec(self, B)
            except torch.cuda.OutOfMemoryError:
                break
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(3):
                e0.record()
                ex.graph.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            ms = float(np.median(ts))
            if B > 1 and ms > 0.5 * slo_ms:           # a batch this large alone takes half the SLO: stop
                ex.close()
                break
            self.ex[B], self.compute_ms[B] = ex, ms
        self.buckets = sorted(self.ex)
        self.ceiling = max(B / (self.compute_ms[B] / 1e3) for B in self.buckets)
        self.gpu_ceiling, self.nic_ceiling = self.ceiling, float("inf")

    def run(self, qps, duration_s, warmup_s=1.0, seed=0, slo_ms=100.0):
        rng = np.random.default_rng(seed)
        n = int(qps * (duration_s + warmup_s))
        arrivals = np.cumsum(rng.exponential(1.0 / qps, size=n))
        queue, inflight = deque(), deque()
        rec, sizes = [], []
        gpu_ms = 0.0
        i = 0
        c0 = thread_cpu()
        t0 = time.perf_counter()
        while i < n or queue or inflight:
            now = time.perf_counter() - t0
            while i < n and arrivals[i] <= now:
                queue.append(arrivals[i])
                i += 1
            while inflight and inflight[0]["done"].query():
                b = inflight.popleft()
                t = time.perf_counter() - t0
                rec.extend((a, t - a, 0) for a in b["arr"])
            while queue and len(inflight) < self.depth:
                t_start = now if not inflight else max(now, inflight[-1]["fin"])
                if inflight and t_start - now > self.lead:
                    break
                B = pow2_floor(min(len(queue), self.buckets[-1]))
                ex = self.ex[B]
                arr = [queue.popleft() for _ in range(B)]
                ex.graph.replay()
                ev = torch.cuda.Event()
                ev.record()
                inflight.append(dict(done=ev, arr=arr, fin=t_start + self.compute_ms[B] / 1e3))
                sizes.append(B)
                gpu_ms += self.compute_ms[B]
        wall = time.perf_counter() - t0
        return self._account(rec, sizes, n, c0, wall, warmup_s, slo_ms, gpu_ms, dict(checks=0, bad_words=0))

    def close(self):
        torch.cuda.synchronize()
        for ex in self.ex.values():
            ex.close()
        self.ex = {}
        torch.cuda.synchronize()


class _RecomputeExec:
    def __init__(self, srv, B, stream=None):
        cfg, model, L, C = srv.cfg, srv.model, srv.L, srv.C
        dev = torch.device("cuda")
        b = Batch([0] * B, [C] * B, new=[L] * B, device=dev)
        items = torch.randint(0, cfg.n_items, (b.total_k,), device=dev)
        acts = torch.randint(0, cfg.n_actions, (b.total_k,), device=dev)

        def fwd():
            return model.forward_full(model.embed(items, acts, b.pos_full), b)
        self._keep = (fwd, items, acts, b)
        st = stream or torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fwd()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        if stream is not None:
            with torch.cuda.graph(self.graph, pool=srv.pool, stream=stream):
                fwd()
        else:
            with torch.cuda.graph(self.graph, pool=srv.pool):
                fwd()
        torch.cuda.synchronize()
        with torch.cuda.stream(st):
            self.graph.replay()                 # first launch now, GPU otherwise idle
        torch.cuda.synchronize()

    def close(self):
        self.graph.reset()
        self._keep = None
