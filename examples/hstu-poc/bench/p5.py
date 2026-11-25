"""P5 driver: latency-throughput curves of the thin serving layer (hstu_poc/serving.py) on 2 nodes.
Strategies: 'fetch' over --transport (gin | raw | verbs; GIN backend by NCCL_GIN_TYPE), plus 'local' (KV already in
GPU memory) and 'recompute' when listed in --strategies. Homogeneous requests: history L, C candidates.

Per offered rate (open-loop Poisson arrivals): p50/p90/p99/mean of request latency (arrival -> scores ready), the
achieved completion rate, mean batch size, and success = the fraction of requests that finished within the SLO
(--slo-ms, default 100 ms) with no data error; goodput = successful requests per second. Fetched batches are
data-checked by sampling (every --check-every-th batch). A sweep stops after two load points with p99 > 3 x SLO or an
achieved rate below 85% of the offered one: latencies far beyond the SLO say nothing new.

-n 8 (srun2.sh): every inference GPU k runs its own server (NIC k, store rank k) at the same offered rate; the
inference ranks meet at a barrier before each load point, so the 8 servers load the node at the same time."""
import argparse
import gc
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
from hstu_poc.serving import Server  # noqa: E402

# users/s per GPU, config M: dense around both saturation points (fetch: 49.4 GB/s / KV bytes per user = 1,473 /
# 736 / 368 / 184 / 92 per s; recompute: ~2,470 / ~1,250 / ~590 / ~250 / ~87 per s at 1k / 2k / 4k / 8k / 16k)
RATES = {1024: [100, 250, 500, 750, 1000, 1150, 1300, 1400, 1500, 1700, 1900, 2100, 2300, 2500, 2700],
         2048: [50, 125, 250, 375, 500, 575, 650, 700, 750, 850, 950, 1050, 1150, 1250, 1350],
         4096: [25, 60, 120, 180, 240, 280, 320, 340, 360, 380, 420, 480, 540, 600, 660],
         8192: [10, 30, 60, 90, 120, 140, 160, 170, 180, 190, 210, 230, 250, 270],
         16384: [5, 15, 30, 45, 60, 70, 75, 80, 85, 90, 95, 100, 110]}


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
    ap.add_argument("--slo-ms", type=float, default=100.0)
    ap.add_argument("--check-every", type=int, default=8, help="data-check every n-th fetched batch (0 = never)")
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
    a = ap.parse_args()
    rank, world, local = hd.init()
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    infer_group = hd.subgroup(range(n))
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
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
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
                                     fetch_stream=fstream, hog=hog, check_every=a.check_every, ext=E, seed=a.seed)
                        ok = 1
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[r{rank}] {name} L={L}: SKIP {str(e)[:80]}", flush=True)
                        srv, ok = None, 0
                    ok = int(hd.allreduce_min(ok, infer_group)) if n > 1 else ok
                    if not ok:
                        if srv is not None:
                            srv.close()
                        continue
                    bad = 0
                    for q in rates:
                        dur = float(min(15.0, max(4.0, 1500.0 / q)))
                        if n > 1:
                            hd.barrier(infer_group)
                        res = srv.run(q, dur, warmup_s=1.0, seed=int(q) + 1000 * rank, slo_ms=a.slo_ms)
                        lat = res.pop("lat")
                        rec = dict(tag=a.tag, config=a.config, strategy=name, L=L, C=a.C, max_batch=mb, gpus=n,
                                   rank=rank, hog_ms=a.hog_ms, hog_period_ms=a.hog_period_ms,
                                   green_fetch=a.green_fetch, max_wait_ms=a.max_wait_ms, offered=q, n=len(lat),
                                   p50_ms=float(np.percentile(lat, 50) * 1e3), p90_ms=float(np.percentile(lat, 90) * 1e3),
                                   p99_ms=float(np.percentile(lat, 99) * 1e3), mean_ms=float(lat.mean() * 1e3),
                                   std_ms=float(lat.std() * 1e3), max_ms=float(lat.max() * 1e3),
                                   cores_other=srv.cpu["other_s"] / srv.cpu["wall_s"],
                                   cores_main=srv.cpu["main_s"] / srv.cpu["wall_s"],
                                   cpu_other_ms_per_req=srv.cpu["other_s"] / srv.cpu["requests"] * 1e3, **res)
                        out.write(json.dumps(rec) + "\n")
                        out.flush()
                        print(f"[r{rank}] {name:9s} L={L:5d} offered {q:6.0f}/s achieved {rec['achieved']:7.1f}/s "
                              f"batch {rec['mean_batch']:5.1f} p50 {rec['p50_ms']:7.2f} p99 {rec['p99_ms']:7.2f} "
                              f"mean {rec['mean_ms']:7.2f} ms; within {a.slo_ms:.0f} ms {100 * rec['success']:5.1f}%; "
                              f"checks {rec['checks']} bad {rec['bad_words']}; cores transport/other "
                              f"{rec['cores_other']:.2f}", flush=True)
                        stop = rec["p99_ms"] > 3 * a.slo_ms or rec["achieved"] < 0.85 * q
                        bad = bad + 1 if stop else 0
                        stop_all = int(hd.allreduce_max(int(bad >= 2), infer_group)) if n > 1 else int(bad >= 2)
                        if stop_all:
                            break
                    srv.close()
                    del srv
                    gc.collect()
                    torch.cuda.empty_cache()
        out.close()
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
