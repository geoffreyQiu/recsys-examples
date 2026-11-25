"""Grid runner for the baselines (P1) and, later, the transports (P2/P3). Study doc §3.1.

One process per GPU. By default each process runs every world-th cell on its own GPU ("1 GPU" numbers, GPUs
independent). With --sync, every process runs the same cell at the same time ("8 GPUs at once": shared host
DRAM, PCIe and NICs). Each cell is one CUDA graph (the whole request: embedding, layers, head, KV movement),
replayed back to back; latency = GPU time per request. Results: JSON lines, one per cell and rank.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from hstu_poc import dist as hd
from hstu_poc.kvsource import H2DKV, LocalKV
from hstu_poc.model import CONFIGS, HSTU, Batch, attn_flops, linear_flops

PEAK_TFLOPS = 989.0   # H100 SXM dense BF16


def hist_lengths(B, L, rng):
    if L != "skew":
        return [int(L)] * B
    # lognormal with median 2k and p99 16k (sigma = ln 8 / 2.326), clipped to [256, 32k], multiples of 64.
    # Drawn from a seed that depends only on B, so every mode and rank sees the same lengths for a cell.
    xs = np.random.default_rng(1000 + B).lognormal(mean=math.log(2048), sigma=math.log(8) / 2.326, size=B)
    return [int(min(max(x, 256), 32768)) // 64 * 64 for x in xs]


def make_source(mode, cfg, extra):
    buf = extra.get("local_buf")
    if mode == "local":
        return LocalKV(cfg, local_buf=buf)
    if mode == "h2d":
        return H2DKV(cfg, local_buf=buf)
    if mode in extra:
        return extra[mode](cfg)
    raise ValueError(mode)


def mem_estimate_gb(cfg, mode, hist, cand):
    tok_kv = cfg.layers * cfg.kv_tok_bytes
    toks = sum(hist) + sum(cand)
    if mode == "recompute":
        return toks * cfg.hidden * 2 * 14 / 1e9 + 2
    q = sum(cand)
    return (toks * tok_kv + q * cfg.hidden * 2 * 14) / 1e9 + 2


def run_cell(model, cfg, mode, hist, cand, args, extra_sources):
    dev = torch.device("cuda")
    B = len(hist)
    src = None
    if mode == "recompute":
        b = Batch([0] * B, cand, new=hist, device=dev)
        T = b.total_k
        items = torch.randint(0, cfg.n_items, (T,), device=dev)
        acts = torch.randint(0, cfg.n_actions, (T,), device=dev)
        flops = cfg.layers * (linear_flops(T, cfg) + attn_flops(b, cfg, recompute=True))

        def fwd():
            x = model.embed(items, acts, b.pos_full)
            return model.forward_full(x, b)
    else:
        b = Batch(hist, cand, device=dev)
        src = make_source(mode, cfg, extra_sources)
        src.setup(b)
        items = torch.randint(0, cfg.n_items, (b.total_q,), device=dev)
        flops = cfg.layers * (linear_flops(b.total_q, cfg) + attn_flops(b, cfg, recompute=False))

        def fwd():
            src.start()
            x = model.embed(items, None, b.pos_q)
            y = model.forward_cached(x, b, src.kv, before_attn=src.before_attn, attn_fn=getattr(src, "attn_fn", None))
            src.end()
            return y

    pre = getattr(src, "pre_replay", None) or (lambda i: None)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            pre(i)
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    if getattr(args, "eager", False):     # P4: no CUDA graph, launch every kernel per request
        class _Eager:
            def replay(self):
                fwd()
        g = _Eager()
    else:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fwd()
    torch.cuda.synchronize()
    # estimate, then measure
    t0 = time.perf_counter()
    for i in range(3):
        pre(i)
        g.replay()
        torch.cuda.synchronize()
    t_est = (time.perf_counter() - t0) / 3
    n = int(min(args.max_iters, max(args.min_iters, args.budget_s / max(t_est, 1e-6))))
    if args.sync:
        n = int(hd.bcast(n))      # all ranks the same count
        hd.barrier()
    # blocking events: the host sleeps instead of spinning while it waits, so process CPU time counts only real
    # work (launches, transport threads such as the verbs worker or NCCL's proxy)
    ev = [(torch.cuda.Event(enable_timing=True, blocking=True), torch.cuda.Event(enable_timing=True, blocking=True))
          for _ in range(n)]
    cpu0 = time.process_time()
    prev = None
    for i, (e0, e1) in enumerate(ev):
        if args.isolated and prev is not None:   # each request alone: request i+1 starts after request i ends
            prev.synchronize()
        e0.record()
        pre(i + 3)
        g.replay()
        e1.record()
        prev = e1
    ev[-1][1].synchronize()
    cpu_s = time.process_time() - cpu0
    ms = np.array([e0.elapsed_time(e1) for e0, e1 in ev])
    moved = src.bytes_moved() if src is not None else 0
    bad = src.check(args.ext, args.seed) if (src is not None and hasattr(src, "check") and args.ext) else None
    res = dict(
        n=n, p50_ms=float(np.percentile(ms, 50)), p90_ms=float(np.percentile(ms, 90)),
        p99_ms=float(np.percentile(ms, 99)), mean_ms=float(ms.mean()), std_ms=float(ms.std(ddof=1)) if n > 1 else 0.0,
        min_ms=float(ms.min()), max_ms=float(ms.max()), users_per_s=B / (ms.mean() / 1e3), hist_tok_per_s=sum(hist) / (ms.mean() / 1e3),
        tflops=flops / (ms.mean() / 1e3) / 1e12, mem_gb=torch.cuda.max_memory_allocated() / 1e9,
        gbs=moved / (ms.mean() / 1e3) / 1e9, bytes=moved, cpu_ms_per_req=cpu_s / n * 1e3, isolated=bool(args.isolated),
    )
    if bad is not None:
        res["check_bad_words"] = bad
    res["eta"] = res["tflops"] / PEAK_TFLOPS
    del g
    if src is not None:
        src.release()
    return res


def add_args(ap):
    ap.add_argument("--configs", default="M")
    ap.add_argument("--modes", default="recompute,local")
    ap.add_argument("--L", default="1024,2048,4096,8192,16384,32768,skew")
    ap.add_argument("--C", default="256")
    ap.add_argument("--B", default="1,2,4,8,16,32,64")
    ap.add_argument("--sync", action="store_true", help="all ranks run the same cell at once")
    ap.add_argument("--budget-s", type=float, default=2.0)
    ap.add_argument("--min-iters", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--mem-gb", type=float, default=70.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--isolated", action="store_true", help="sync before each request (no overlap between requests)")
    ap.add_argument("--eager", action="store_true", help="no CUDA graph (P4 baseline)")


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    a.ext, a.seed = None, 0
    rank, world, local = hd.init() if a.sync else (int(os.environ.get("SLURM_PROCID", 0)),
                                                   int(os.environ.get("SLURM_NTASKS", 1)),
                                                   int(os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local)
    run_grid(a, rank, world, {})


def run_grid(a, rank, world, extra, gpus=None):
    """Run the cells of `a` on this rank. extra: {mode: factory(cfg) -> source} for transport modes."""
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cells = [(c, m, int(B), L, int(C)) for c in a.configs.split(",") for C in a.C.split(",")
             for L in a.L.split(",") for B in a.B.split(",") for m in a.modes.split(",")]
    mine = cells if a.sync else cells[rank::world]
    out = open(f"{a.out}.rank{rank}.jsonl", "a")
    models = {}
    rng = np.random.default_rng(1234)
    with torch.inference_mode():
        for (c, mode, B, L, C) in mine:
            cfg = CONFIGS[c]
            if c not in models:
                models.clear()
                torch.cuda.empty_cache()
                models[c] = HSTU(cfg)
            hist = hist_lengths(B, L, rng)
            cand = [C] * B
            rec = dict(tag=a.tag, config=c, mode=mode, B=B, L=L, C=C, hist_mean=float(np.mean(hist)),
                       hist_max=max(hist), gpus=gpus or (world if a.sync else 1), rank=rank)
            est = mem_estimate_gb(cfg, mode, hist, cand)
            if est > a.mem_gb:
                rec.update(skipped=f"memory estimate {est:.0f} GB > {a.mem_gb:.0f}")
            else:
                torch.cuda.reset_peak_memory_stats()
                try:
                    rec.update(run_cell(models[c], cfg, mode, hist, cand, a, extra))
                except torch.cuda.OutOfMemoryError as e:
                    rec.update(skipped=f"OOM: {str(e)[:80]}")
                torch.cuda.empty_cache()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if "skipped" in rec:
                print(f"[r{rank}] {c} {mode:9s} B={B:3d} L={L:>5} C={C:4d}: SKIP {rec['skipped']}", flush=True)
            else:
                print(f"[r{rank}] {c} {mode:9s} B={B:3d} L={L:>5} C={C:4d}: mean {rec['mean_ms']:8.3f} "
                      f"(sd {rec['std_ms']:.3f}) p50 {rec['p50_ms']:8.3f} ms p99 "
                      f"{rec['p99_ms']:8.3f} users/s {rec['users_per_s']:9.1f} TFLOPS {rec['tflops']:6.1f} "
                      f"GB/s {rec['gbs']:6.1f} cpu {rec['cpu_ms_per_req']:.3f} ms/req n={rec['n']}"
                      + (f" check {rec['check_bad_words']}" if "check_bad_words" in rec else ""), flush=True)
    out.close()


if __name__ == "__main__":
    main()
