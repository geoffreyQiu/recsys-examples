"""P5 driver, v1 (the serving loop as of the first P5 runs; kept to reproduce the serving crash): latency-throughput curves of the thin serving layer (hstu_poc/serving.py) on 2 nodes.
Strategies: 'fetch' over --transport (gin | raw | verbs; GIN backend by NCCL_GIN_TYPE), plus 'local' (KV already in
GPU memory) and 'recompute' when listed in --strategies. Homogeneous requests: history L, C candidates.
For each offered rate: p50/p90/p99 of request latency (arrival -> scores ready), achieved rate, mean batch size."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2 import NICS, WORKER_CORES, fill_pool  # noqa: E402

from hstu_poc import dist as hd  # noqa: E402
from hstu_poc.ext import ext  # noqa: E402
from hstu_poc.kvsource import GinKVSource, RawGdakiSource, VerbsKVSource  # noqa: E402
from hstu_poc.model import CONFIGS, HSTU  # noqa: E402
from hstu_poc.serving_v1 import Server  # noqa: E402  (old serving loop, crash reproduction)

RATES = {1024: [200, 400, 800, 1200, 1400, 1600, 2000, 2400, 3200, 4800, 8000],
         2048: [100, 200, 400, 600, 700, 800, 1000, 1200, 1600, 2400, 4000],
         4096: [50, 100, 200, 300, 350, 400, 450, 550, 700, 1000, 1600, 3200],
         16384: [10, 20, 40, 60, 80, 90, 100, 120, 160, 240, 400, 800]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True, choices=["gin", "raw", "verbs"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--strategies", default="fetch")
    ap.add_argument("--config", default="M")
    ap.add_argument("--L", default="4096")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-wait-ms", type=float, default=2.0)
    ap.add_argument("--rates", default=None, help="comma list; default per L")
    ap.add_argument("--pool-gb", type=float, default=40)
    ap.add_argument("--local-gb", type=float, default=40)
    ap.add_argument("--nctx", type=int, default=4)
    ap.add_argument("--nqp", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="p5")
    ap.add_argument("--hog-ms", type=float, default=0, help="P6: co-located job occupying every SM slot for this long")
    ap.add_argument("--hog-period-ms", type=float, default=10)
    ap.add_argument("--green-fetch", type=int, default=0, help="P6: posts on a green context with this many SMs "
                    "(the hog then runs on the rest)")
    ap.add_argument("--gc-mode", default="none", choices=["none", "idle", "midrun"],
                    help="crash experiment: when the previous Server (kept alive by a reference cycle) is collected")
    ap.add_argument("--gc-at", type=int, default=100, help="midrun: batch index of the forced collection")
    ap.add_argument("--sanity", action="store_true", help="crash hunt: replay every executor graph once after "
                    "building a Server (synchronized, logged)")
    ap.add_argument("--no-emptycache", action="store_true", help="crash hunt: make torch.cuda.empty_cache a no-op "
                    "(PyTorch calls it before every graph capture)")
    a = ap.parse_args()
    if a.no_emptycache:
        torch.cuda.empty_cache = lambda: None
    n_servers = 0
    if a.gc_mode == "midrun":         # keep a dropped Server alive until the forced collection inside the next run
        import gc
        gc.disable()
    rank, world, local = hd.init()
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    E = ext()
    label = a.label or a.transport
    pool = int(a.pool_gb * 2**30)
    if a.transport == "gin":
        uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
        nbytes = int(max(a.local_gb, a.pool_gb)) * 2**30
        pool = nbytes
        tr = E.GinKV(uid, rank, world, nbytes, not infer, n_ctx=a.nctx + 2, queue_depth=4096, strong_signals=True,
                     n_signals=4)
        make = lambda cfg: GinKVSource(cfg, tr, E, peer, pool, n_ctx=a.nctx).p4_init("5a")  # noqa: E731
    elif a.transport == "raw":
        tr = (E.RawGdaki(NICS[local], int(a.local_gb * 2**30), a.nqp, 4096) if infer
              else E.VerbsKV(NICS[local], pool, False, a.nqp, 64, 0, -1))
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: RawGdakiSource(cfg, tr, E, pool).p4_init("5a")  # noqa: E731
    else:
        tr = E.VerbsKV(NICS[local], int(a.local_gb * 2**30) if infer else pool, infer, a.nqp, 64, 0, WORKER_CORES[local])
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: VerbsKVSource(cfg, tr, E, pool).p4_init("5a")  # noqa: E731
    if not infer:
        fill_pool(E, tr.local(), a.seed)
    hd.barrier()
    if infer:
        cfg = CONFIGS[a.config]
        model = HSTU(cfg)
        out = open(f"{a.out}.rank{rank}.jsonl", "a")
        with torch.inference_mode():
            for L in map(int, a.L.split(",")):
                rates = list(map(float, a.rates.split(","))) if a.rates else RATES[L]
                for strat in a.strategies.split(","):
                    name = label if strat == "fetch" else strat
                    mb = a.max_batch if strat != "recompute" else min(a.max_batch, max(1, 65536 // L * 2))
                    fstream, hog = None, None
                    if a.green_fetch:
                        g_small, g_rest, n_small, n_rest = E.green_streams(a.green_fetch)
                        fstream = torch.cuda.ExternalStream(g_small)
                    if a.hog_ms:
                        hs = torch.cuda.ExternalStream(g_rest) if a.green_fetch else torch.cuda.Stream()
                        hog = (E, a.hog_ms, a.hog_period_ms, hs)
                    try:
                        srv = Server(model, cfg, L, a.C, strat, make_source=make, window=tr.local(),
                                     max_batch=mb, max_wait_ms=a.max_wait_ms, regions=2 if strat == "fetch" else 1,
                                     fetch_stream=fstream, hog=hog)
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[r{rank}] {name} L={L}: SKIP {str(e)[:80]}", flush=True)
                        continue
                    if a.sanity:
                        for k_, ((B_, r_), ex_) in enumerate(sorted(srv.ex.items())):
                            print(f"sanity L={L} B={B_} region={r_}", flush=True)
                            ex_.launch(k_)
                            torch.cuda.synchronize()
                        print(f"sanity L={L}: all {len(srv.ex)} graphs ok", flush=True)
                    saturated = 0
                    n_servers += 1
                    for qi, q in enumerate(rates):
                        dur = float(min(12.0, max(3.0, 400.0 / q)))
                        gc_at = a.gc_at if (a.gc_mode == "midrun" and n_servers > 1 and qi == 0) else None
                        lat, rate, mean_b = srv.run(q, dur, warmup_s=1.0, seed=int(q), gc_at=gc_at)
                        rec = dict(tag=a.tag, config=a.config, strategy=name, L=L, C=a.C, max_batch=mb,
                                   hog_ms=a.hog_ms, hog_period_ms=a.hog_period_ms, green_fetch=a.green_fetch,
                                   max_wait_ms=a.max_wait_ms, offered=q, achieved=rate, n=len(lat), mean_batch=mean_b,
                                   p50_ms=float(np.percentile(lat, 50) * 1e3), p90_ms=float(np.percentile(lat, 90) * 1e3),
                                   p99_ms=float(np.percentile(lat, 99) * 1e3), mean_ms=float(lat.mean() * 1e3),
                                   cores_other=srv.cpu["other_s"] / srv.cpu["wall_s"],
                                   cores_main=srv.cpu["main_s"] / srv.cpu["wall_s"],
                                   cpu_other_ms_per_req=srv.cpu["other_s"] / srv.cpu["requests"] * 1e3)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        print(f"[r{rank}] {name:9s} L={L:5d} offered {q:7.0f}/s achieved {rate:7.1f}/s batch {mean_b:5.1f} "
                              f"p50 {rec['p50_ms']:8.2f} p90 {rec['p90_ms']:8.2f} p99 {rec['p99_ms']:8.2f} ms; cores: transport/other "
                              f"{rec['cores_other']:.2f}, serving loop {rec['cores_main']:.2f}", flush=True)
                        if rate < 0.85 * q or rec["p99_ms"] > 3000:
                            saturated += 1
                            if saturated >= 2:
                                break
                    del srv
                    if a.gc_mode == "idle":       # collect the Server <-> Executor cycle now, with the GPU idle
                        import gc
                        torch.cuda.synchronize()
                        print(f"gc.collect() between servers: {gc.collect()} objects", flush=True)
                        torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        out.close()
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
