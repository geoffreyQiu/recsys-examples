"""P5: a thin serving layer (no Triton). Study doc §10.

- Open-loop load: Poisson arrivals at a fixed offered rate; homogeneous requests (same history L and candidates C).
- Dynamic batcher: batch sizes are powers of two up to max_batch (rounded down, never padded); a batch forms when a
  KV region is free and (the GPU has nothing in flight, or max_batch requests wait, or the oldest waited max_wait).
- Fetch at admission: a batch's KV transfer is posted when the batch forms, into one of two ping-pong regions of
  the local window, so batch k+1's transfer overlaps batch k's compute. GIN / raw GDAKI: the post kernel runs on a
  fetch stream; CPU verbs: the worker thread posts. The compute graph (per batch size and region) waits per layer.
- Strategies: fetch (a remote transport), local (KV already in GPU memory: the upper bound), recompute (no cache).
- Per request: completion time - arrival time (host clock; completion = the batch's done event observed by a
  polling loop, ~10 us granularity).
"""
import os
import threading
import time

import numpy as np
import torch

from .kvsource import LocalKV, VerbsKVSource
from .model import Batch

_SEQ = {}   # CPU verbs: one sequence counter per engine (its flag array is shared by every region and batch size)


def thread_cpu():
    """CPU seconds per thread of this process (utime + stime from /proc)."""
    out, hz = {}, os.sysconf("SC_CLK_TCK")
    for t in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{t}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            out[int(t)] = (int(parts[11]) + int(parts[12])) / hz
        except OSError:
            pass
    return out


class Executor:
    """One CUDA graph for (batch size, region)."""

    def __init__(self, server, B, region):
        s = server
        cfg, model, L, C = s.cfg, s.model, s.L, s.C
        self.B, self.region, self.server = B, region, s
        dev = torch.device("cuda")
        self.src = None
        if s.strategy == "recompute":
            b = Batch([0] * B, [C] * B, new=[L] * B, device=dev)
            items = torch.randint(0, cfg.n_items, (b.total_k,), device=dev)
            acts = torch.randint(0, cfg.n_actions, (b.total_k,), device=dev)

            def fwd():
                return model.forward_full(model.embed(items, acts, b.pos_full), b)
        else:
            b = Batch([L] * B, [C] * B, device=dev)
            if s.strategy == "local":
                src = LocalKV(cfg, local_buf=s.window[region * s.region_bytes:(region + 1) * s.region_bytes])
            else:
                src = s.make_source(cfg)
                src.base = region * s.region_bytes
                src.post_outside = True
            src.setup(b)
            self.src = src
            items = torch.randint(0, cfg.n_items, (b.total_q,), device=dev)

            def fwd():
                if not isinstance(src, VerbsKVSource):
                    src.start()
                x = model.embed(items, None, b.pos_q)
                y = model.forward_cached(x, b, src.kv, before_attn=src.before_attn)
                src.end()
                return y
        self.b = b
        # the graph captures these tensors' addresses: keep them alive as long as the executor
        self._keep = (fwd, items, locals().get("acts"))
        self.done = torch.cuda.Event()
        self.posted = torch.cuda.Event()
        # warm-up (eager, after a real transfer), then capture the compute graph (the post stays outside)
        self._post(0)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        st.wait_event(self.posted)
        with torch.cuda.stream(st):
            fwd()
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=s.pool):   # graphs replay one at a time: one shared pool
            fwd()
        torch.cuda.synchronize()

    def _post(self, i):
        """Start this batch's KV transfer now (fetch at admission). Never waits for earlier batches' compute: the
        region is free (the CPU only reuses a region after its previous batch's done event)."""
        src, s = self.src, self.server
        main = torch.cuda.current_stream()
        if src is None or s.strategy == "local":
            self.posted.record(main)
            return
        if isinstance(src, VerbsKVSource):
            key = id(src.v)
            _SEQ[key] = _SEQ.get(key, 0) + 1
            src.seq = _SEQ[key]
            src.cur = i % src.variants
            src.seq_dev.fill_(src.seq)              # main stream: ordered before this batch's waits
            src.v.post_reads(src.tables[src.cur], src.layer_ptr_cpu, src.seq)
            self.posted.record(main)
        else:
            fs = s.fetch_stream
            with torch.cuda.stream(fs):
                src.pre_replay(i)                   # this request's users (D2D copy of the plan) + post kernel
                src.post()
                self.posted.record(fs)

    def launch(self, i):
        self._post(i)
        torch.cuda.current_stream().wait_event(self.posted)
        self.graph.replay()
        self.done.record()


class Server:
    def __init__(self, model, cfg, L, C, strategy, make_source=None, window=None, max_batch=32, max_wait_ms=2.0,
                 regions=2, fetch_stream=None, hog=None):
        """fetch_stream: stream for fetch-at-admission posts (P6: a green-context stream); hog: (ext, ms, period_ms,
        stream) = a co-located job that fills every SM slot for ms every period_ms (P6)."""
        self.model, self.cfg, self.L, self.C = model, cfg, L, C
        self.strategy, self.make_source, self.window = strategy, make_source, window
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1e3
        self.regions = regions
        self.buckets = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= max_batch]
        per_req = cfg.layers * 2 * (L + C) * cfg.hd * 2
        self.region_bytes = (per_req * max_batch + (1 << 21)) // (1 << 21) * (1 << 21) if window is not None else 0
        if window is not None and regions * self.region_bytes > window.numel():
            raise torch.cuda.OutOfMemoryError("window too small for the regions")
        self.fetch_stream = fetch_stream or torch.cuda.Stream()
        self.hog = hog
        self.pool = torch.cuda.graph_pool_handle()
        self.ex = {}
        for B in self.buckets:
            for r in range(regions):
                self.ex[(B, r)] = Executor(self, B, r)

    def run(self, qps, duration_s, warmup_s=1.0, seed=0, gc_at=None):
        """Open-loop run. Returns latencies (s) of requests that arrived after the warm-up, the achieved completion
        rate over the measured span, and the number of batches. gc_at: run Python's cycle collector right after
        launching that batch (crash experiment: collects a previous Server that is still alive in a cycle)."""
        rng = np.random.default_rng(seed)
        n = int(qps * (duration_s + warmup_s))
        arrivals = np.cumsum(rng.exponential(1.0 / qps, size=n))
        queue = []                      # arrival times of waiting requests
        inflight = []                   # (executor, region, [arrival times])
        free = list(range(self.regions))
        rec = []                        # (arrival, latency)
        i = nb = 0
        sizes = []
        c0 = thread_cpu()
        t0 = time.perf_counter()
        next_hog = 0.0
        n_sm = torch.cuda.get_device_properties(0).multi_processor_count
        while i < n or queue or inflight:
            now = time.perf_counter() - t0
            if self.hog is not None and now >= next_hog:      # P6: co-located job, all SM slots for ms
                E, hog_ms, period_ms, hog_stream = self.hog
                with torch.cuda.stream(hog_stream):
                    E.hog(int(hog_ms * 1e6), n_sm * 16, 128)
                next_hog = now + period_ms / 1e3
            while i < n and arrivals[i] <= now:
                queue.append(arrivals[i])
                i += 1
            if inflight:
                still = []
                for ex, r, arr in inflight:
                    if ex.done.query():
                        t = time.perf_counter() - t0
                        rec.extend((a, t - a) for a in arr)
                        free.append(r)
                    else:
                        still.append((ex, r, arr))
                inflight = still
            if queue and free and (not inflight or len(queue) >= self.max_batch or now - queue[0] >= self.max_wait):
                B = max(b for b in self.buckets if b <= len(queue))
                r = free.pop(0)
                ex = self.ex[(B, r)]
                arr, queue = queue[:B], queue[B:]
                ex.launch(nb)
                nb += 1
                if gc_at is not None and nb == gc_at:
                    import gc
                    print(f"gc.collect() at batch {nb}: {gc.collect()} objects", flush=True)
                sizes.append(B)
                inflight.append((ex, r, arr))
        wall = time.perf_counter() - t0
        c1 = thread_cpu()
        me = threading.get_native_id()
        self.cpu = dict(wall_s=wall, main_s=c1.get(me, 0) - c0.get(me, 0), requests=n,
                        other_s=sum(v - c0.get(t, 0) for t, v in c1.items() if t != me))
        rec = np.array(rec)
        kept = rec[rec[:, 0] >= warmup_s]
        span = float((kept[:, 0] + kept[:, 1]).max() - warmup_s)
        return kept[:, 1], len(kept) / max(span, 1e-9), float(np.mean(sizes))
