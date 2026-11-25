"""P3 driver: workload W (prefix P local, S new tokens computed and written to the remote store, C candidates) on
2 nodes (scripts/srun2.sh -n N -- python bench/p3.py --transport gin|verbs ...). Modes per cell:
  none      compute only (the same forward, no writes): the bar for "added latency"
  <label>   the same forward with per-layer writes of the new tokens' K/V (6') and a durability fence
Per request: scores-ready latency (t1 - t0) and writes-durable latency (t2 - t0), from GPU timestamps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2 import NICS, WORKER_CORES, fill_pool  # noqa: E402

from hstu_poc import dist as hd  # noqa: E402
from hstu_poc.ext import ext  # noqa: E402
from hstu_poc.kvsource import LocalKV  # noqa: E402
from hstu_poc.model import CONFIGS, HSTU, Batch, attn_flops, linear_flops  # noqa: E402
from hstu_poc.writers import GinWriter, NoWrite, VerbsWriter  # noqa: E402


def cells(a):
    out = []
    for c in a.configs.split(","):
        for B in map(int, a.B.split(",")):
            for C in map(int, a.C.split(",")):
                for S in [int(x) for x in a.fresh.split(",") if x]:   # fresh sequences: P = 0, S = L
                    out.append((c, B, 0, S, C))
                for P in [int(x) for x in a.P.split(",") if x]:
                    for S in [int(x) for x in a.S.split(",") if x]:
                        out.append((c, B, P, S, C))
    return out


def run_cell(model, cfg, writer, B, P, S, C, a, E, window):
    dev = torch.device("cuda")
    b = Batch([P] * B, [C] * B, new=[S] * B, device=dev)
    src = LocalKV(cfg, local_buf=window)
    src.setup(b)                                    # prefix rows: staged before timing (any values)
    writer.setup(b, src.blk)
    items = torch.randint(0, cfg.n_items, (b.total_q,), device=dev)
    acts = torch.randint(0, cfg.n_actions, (b.total_q,), device=dev)
    stamps = torch.zeros(3, dtype=torch.int64, device=dev)

    def fwd():
        E.stamp(stamps[0:1])
        writer.start()
        x = model.embed(items, acts, b.pos_q)
        model.forward_cached(x, b, src.kv, after_kv=writer.after_kv)
        E.stamp(stamps[1:2])
        writer.end(stamps)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            writer.pre_replay(i)
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(3):
        writer.pre_replay(2 + i)
        g.replay()
        torch.cuda.synchronize()
    t_est = (time.perf_counter() - t0) / 3
    n = int(min(a.max_iters, max(a.min_iters, a.budget_s / max(t_est, 1e-6))))
    if a.sync:
        n = int(hd.bcast(n))
        hd.barrier()
    hist = torch.zeros(n, 3, dtype=torch.int64, device=dev)
    ev = [torch.cuda.Event(blocking=True) for _ in range(n)]
    cpu0 = time.process_time()
    for i in range(n):
        if i:
            ev[i - 1].synchronize()
        writer.pre_replay(5 + i)
        g.replay()
        hist[i].copy_(stamps)
        ev[i].record()
    ev[-1].synchronize()
    cpu_s = time.process_time() - cpu0
    h = hist.cpu().numpy().astype(np.float64)
    scores = (h[:, 1] - h[:, 0]) / 1e6
    durable = (h[:, 2] - h[:, 0]) / 1e6
    bad = writer.readback(E)
    flops = cfg.layers * (linear_flops(b.total_q, cfg) + attn_flops(b, cfg, recompute=False))
    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0  # noqa: E731
    res = dict(n=n, scores_p50=float(np.percentile(scores, 50)), scores_p99=float(np.percentile(scores, 99)),
               durable_p50=float(np.percentile(durable, 50)), durable_p99=float(np.percentile(durable, 99)),
               scores_mean=float(scores.mean()), scores_std=sd(scores), durable_mean=float(durable.mean()),
               durable_std=sd(durable), tail_mean=float((durable - scores).mean()),
               bytes=writer.bytes, write_gbs=writer.bytes / (np.median(durable) / 1e3) / 1e9 if writer.bytes else 0.0,
               cpu_ms_per_req=cpu_s / n * 1e3, tflops=flops / (np.median(scores) / 1e3) / 1e12,
               mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if bad is not None:
        res["readback_bad_bytes"] = bad
    del g
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True, choices=["gin", "verbs"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--modes", default="none,gin")
    ap.add_argument("--configs", default="M")
    ap.add_argument("--B", default="1,8")
    ap.add_argument("--C", default="0,256")
    ap.add_argument("--fresh", default="2048,4096,8192,16384", help="S values with P = 0")
    ap.add_argument("--P", default="2048,16384")
    ap.add_argument("--S", default="16,64,256,1024")
    ap.add_argument("--pool-gb", type=float, default=40)
    ap.add_argument("--local-gb", type=float, default=40)
    ap.add_argument("--nctx", type=int, default=2, help="GIN write contexts")
    ap.add_argument("--nqp", type=int, default=2)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--budget-s", type=float, default=1.0)
    ap.add_argument("--min-iters", type=int, default=10)
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rank, world, local = hd.init()
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    infer_group = hd.subgroup(range(n))
    E = ext()
    label = a.label or a.transport
    if a.transport == "gin":
        uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
        nbytes = int(max(a.local_gb, a.pool_gb)) * 2**30
        pool = nbytes
        tr = E.GinKV(uid, rank, world, nbytes, not infer, n_ctx=4 + a.nctx, queue_depth=2048, strong_signals=True,
                     n_signals=4)
        make = lambda cfg: GinWriter(cfg, tr, E, peer, pool, ctx0=4, n_ctx=a.nctx)  # noqa: E731
    else:
        pool = int(a.pool_gb * 2**30)
        nbytes = int(a.local_gb * 2**30) if infer else pool
        tr = E.VerbsKV(NICS[local], nbytes, infer, a.nqp, 64, 0, WORKER_CORES[local])
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: VerbsWriter(cfg, tr, E, pool)  # noqa: E731
    if not infer:
        fill_pool(E, tr.local(), a.seed)
    hd.barrier()
    if infer:
        hd.GROUP = infer_group
        window = tr.local()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out = open(f"{a.out}.rank{rank}.jsonl", "a")
        models = {}
        with torch.inference_mode():
            for (c, B, P, S, C) in cells(a):
                cfg = CONFIGS[c]
                if c not in models:
                    models.clear()
                    models[c] = HSTU(cfg)
                for mode in a.modes.split(","):
                    writer = NoWrite(cfg, window, pool) if mode == "none" else make(cfg)
                    rec = dict(tag=a.tag, config=c, mode=mode if mode == "none" else label, B=B, P=P, S=S, C=C,
                               gpus=n, rank=rank)
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        rec.update(run_cell(models[c], cfg, writer, B, P, S, C, a, E, window))
                    except torch.cuda.OutOfMemoryError as e:
                        rec.update(skipped=str(e)[:100])
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if "skipped" in rec:
                        print(f"[r{rank}] {c} {rec['mode']:6s} B={B:3d} P={P:5d} S={S:5d} C={C:4d}: SKIP {rec['skipped']}", flush=True)
                    else:
                        print(f"[r{rank}] {c} {rec['mode']:6s} B={B:3d} P={P:5d} S={S:5d} C={C:4d}: scores mean "
                              f"{rec['scores_mean']:8.3f} ms, durable mean {rec['durable_mean']:8.3f} p99 {rec['durable_p99']:8.3f}"
                              f" ms, write {rec['write_gbs']:5.1f} GB/s, cpu {rec['cpu_ms_per_req']:.3f} ms/req"
                              + (f", readback bad {rec['readback_bad_bytes']}" if "readback_bad_bytes" in rec else ""),
                              flush=True)
        out.close()
        hd.GROUP = None
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
