"""P5s: serving throughput within a latency SLO, each strategy on its own with the overlap-optimized servers
(hstu_poc/serving2.py). One process per (strategy, config, history length):
  fetch:     scripts/srun2.sh -j J -n 1 -- python bench/p5s.py --strategy fetch --transport raw|gin|verbs ...
  recompute: scripts/srun2.sh -j J -1 -- python bench/p5s.py --strategy recompute ...
Offered loads are fractions of the strategy's ceiling (fetch: min(NIC rate / KV bytes per user, compute at the best
batch size); recompute: compute at the best batch size within half the SLO). Per load point (open-loop Poisson
arrivals): latency p50/p90/p99/mean (arrival -> output ready), achieved rate, success (latency within the SLO and no
data error), mean batch size, GPU compute share, CPU cores of other threads (transport). A sweep stops after two
failing points (p99 > SLO, success < 99%, or achieved < 90% of offered: a 5 s Poisson sample varies by a few %).
A point runs for --min-requests requests (5 s .. --max-dur s)."""
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
from hstu_poc.model import CONFIGS, HSTU, Batch  # noqa: E402
from hstu_poc.serving2 import Cotenant, FetchServer, RecomputeServer  # noqa: E402

FRACS = "0.3,0.5,0.65,0.75,0.82,0.88,0.92,0.95,0.98,1.01,1.05"


def selftest(srv, model):
    """The graph re-pointed at slots near the ring's end must give the same output as a plain forward over a
    contiguous copy of the same K/V (the cu_seqlens_k / row-index rewrite is exact)."""
    B = 2
    ex = srv.ex[B]
    s0 = srv.n_slots - B
    srv._post_range(s0, B)
    ex.launch(s0, s0 + B - 1)
    torch.cuda.synchronize()
    y_ring = ex.out.clone()
    R = srv.ring.R
    kv_ref = [(k[s0 * R:(s0 + B) * R].clone(), v[s0 * R:(s0 + B) * R].clone()) for k, v in srv.ring.views]
    b = Batch([srv.L] * B, [srv.C] * B)
    y_ref = model.forward_cached(model.embed(ex.items, None, b.pos_q), b, lambda l: kv_ref[l])
    torch.cuda.synchronize()
    return float((y_ring.float() - y_ref.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["fetch", "recompute"])
    ap.add_argument("--transport", default="raw", choices=["raw", "gin", "verbs"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--config", default="M")
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--slo-ms", type=float, default=100.0)
    ap.add_argument("--fracs", default=FRACS)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--pool-gb", type=float, default=40)
    ap.add_argument("--local-gb", type=float, default=40)
    ap.add_argument("--nctx", type=int, default=4)
    ap.add_argument("--nqp", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="p5s")
    ap.add_argument("--min-requests", type=int, default=2000, help="run each load point for about this many "
                    "requests (5 s at least, --max-dur at most)")
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--p4", default="5a", choices=["5a", "fused"], help="fetch: per-layer wait variant")
    ap.add_argument("--tuned", action="store_true", help="CPU verbs: spinning worker and inline posting")
    ap.add_argument("--stop", default="slo", choices=["slo", "sat"], help="slo: stop after two points failing the SLO; "
                    "sat: keep going until two points are past saturation (achieved < 97%% of offered or p99 > "
                    "--sat-p99-ms), for capacities at relaxed SLOs")
    ap.add_argument("--sat-p99-ms", type=float, default=2000.0)
    ap.add_argument("--cotenant", default=None, help="P6: a co-located recompute job 'LxB' (history L, B users) kept "
                    "busy on its own stream while the fetch server runs")
    ap.add_argument("--green", type=int, default=0, help="P6: confine the co-tenant to a green context without N SMs "
                    "and post GDAKI transfers from an N-SM one")
    ap.add_argument("--priority", action="store_true", help="P6: run the server's compute and posts on high-priority "
                    "streams (the co-tenant stays at normal priority)")
    a = ap.parse_args()
    rank, world, local = hd.init()
    cfg = CONFIGS[a.config]
    E = ext()
    tr, peer, pool, infer = None, None, int(a.pool_gb * 2**30), True
    if a.strategy == "fetch":
        n = world // 2
        infer = rank < n
        peer = rank + n if infer else rank - n
        if a.transport == "gin":
            uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
            nbytes = int(max(a.local_gb, a.pool_gb)) * 2**30
            pool = nbytes
            tr = E.GinKV(uid, rank, world, nbytes, not infer, n_ctx=a.nctx + 2, queue_depth=4096,
                         strong_signals=True, n_signals=4)
        elif a.transport == "raw":
            tr = (E.RawGdaki(NICS[local], int(a.local_gb * 2**30), a.nqp, 4096) if infer
                  else E.VerbsKV(NICS[local], pool, False, a.nqp, 64, 0, -1))
            infos = hd.gather(tr.local_info())
            tr.connect(infos[peer])
        else:
            tr = E.VerbsKV(NICS[local], int(a.local_gb * 2**30) if infer else pool, infer, a.nqp, 64, 0,
                           WORKER_CORES[local])
            infos = hd.gather(tr.local_info())
            tr.connect(infos[peer])
            if infer and a.tuned:
                tr.set_spin(True)
                tr.set_inline_post(True)
        if not infer:
            fill_pool(E, tr.local(), a.seed)
        hd.barrier()
    label = a.label or (a.transport if a.strategy == "fetch" else "recompute")
    if infer:
        model = HSTU(cfg)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out = open(f"{a.out}.rank{rank}.jsonl", "a")
        with torch.inference_mode():
            fstream, cot, cot_alone, cot_ms = None, None, None, None
            hp = None
            if a.priority:
                lo, hi = torch.cuda.Stream.priority_range()
                hp = torch.cuda.Stream(priority=hi)
                fstream = torch.cuda.Stream(priority=hi)
            if a.green:
                g_small, g_rest, n_small, n_rest = E.green_streams(a.green)
                fstream = torch.cuda.ExternalStream(g_small)
            try:
                if a.strategy == "fetch":
                    srv = FetchServer(model, cfg, a.L, a.C, a.transport, tr, E, tr.local(), pool, peer=peer,
                                      n_ctx=a.nctx, max_batch=a.max_batch, depth=a.depth, chunks=a.chunks,
                                      seed=a.seed, fetch_stream=fstream, p4=a.p4)
                    if a.cotenant:
                        Lc, Bc = map(int, a.cotenant.split("x"))
                        cot = Cotenant(model, cfg, Lc, Bc,
                                       stream=torch.cuda.ExternalStream(g_rest) if a.green else None)
                        cot_alone, cot_ms = cot.rate_alone(), cot.unit_ms
                        print(f"[r{rank}] co-tenant recompute {a.config} {Lc}x{Bc}: {cot_ms:.2f} ms per batch, "
                              f"{cot_alone:.1f} batches/s alone{' (green: ' + str(n_rest) + ' SMs)' if a.green else ''}",
                              flush=True)
                    err = selftest(srv, model)
                    print(f"[r{rank}] {label} {a.config} L={a.L}: ring self-test max |diff| {err:.3g} "
                          f"({'PASS' if err == 0 else 'FAIL'}); {srv.n_slots} slots", flush=True)
                else:
                    srv = RecomputeServer(model, cfg, a.L, a.C, max_batch=a.max_batch, depth=a.depth,
                                          slo_ms=a.slo_ms)
                    err = None
            except torch.cuda.OutOfMemoryError as e:
                print(f"[r{rank}] {label} {a.config} L={a.L}: SKIP {str(e)[:100]}", flush=True)
                srv = None
            if srv is not None:
                cms = " ".join(f"B{B}:{ms:.2f}" for B, ms in sorted(srv.compute_ms.items()))
                print(f"[r{rank}] {label} {a.config} L={a.L}: compute ms {cms}; ceiling {srv.ceiling:,.0f}/s "
                      f"(NIC {srv.nic_ceiling:,.0f}, GPU {srv.gpu_ceiling:,.0f})", flush=True)
                bad = 0
                for f in map(float, a.fracs.split(",")):
                    q = f * srv.ceiling
                    dur = float(min(a.max_dur, max(5.0, a.min_requests / q)))
                    with torch.cuda.stream(hp) if hp is not None else torch.cuda.stream(torch.cuda.current_stream()):
                        if cot is not None:
                            res = srv.run(q, dur, warmup_s=1.0, seed=int(q * 10) + 7, slo_ms=a.slo_ms, cotenant=cot)
                        else:
                            res = srv.run(q, dur, warmup_s=1.0, seed=int(q * 10) + 7, slo_ms=a.slo_ms)
                    lat = res.pop("lat")
                    # relaxed SLOs: share of requests within each latency target, the tail, and drift (p99 of the
                    # first vs the second half of the run: a growing queue = past saturation)
                    half = max(len(lat) // 2, 1)
                    res.update(succ_at={str(x): float((lat * 1e3 <= x).mean()) for x in (100, 200, 500, 1000, 2000)},
                               p999_ms=float(np.percentile(lat, 99.9) * 1e3),
                               p99_h1_ms=float(np.percentile(lat[:half], 99) * 1e3),
                               p99_h2_ms=float(np.percentile(lat[half:], 99) * 1e3) if len(lat) > half else None,
                               p4=a.p4 if a.strategy == "fetch" else None, tuned=a.tuned, stop=a.stop,
                               slo_cap_ms=a.slo_ms)
                    rec = dict(tag=a.tag, config=a.config, strategy=label, kind=a.strategy, L=a.L, C=a.C,
                               depth=a.depth, chunks=a.chunks, frac=f, offered=q,
                               ceiling=srv.ceiling, nic_ceiling=srv.nic_ceiling, gpu_ceiling=srv.gpu_ceiling,
                               compute_ms=srv.compute_ms, selftest=err, n=len(lat),
                               p50_ms=float(np.percentile(lat, 50) * 1e3), p90_ms=float(np.percentile(lat, 90) * 1e3),
                               p99_ms=float(np.percentile(lat, 99) * 1e3), mean_ms=float(lat.mean() * 1e3),
                               max_ms=float(lat.max() * 1e3), cores_other=srv.cpu["other_s"] / srv.cpu["wall_s"],
                               cores_main=srv.cpu["main_s"] / srv.cpu["wall_s"], cotenant=a.cotenant, green=a.green,
                               priority=a.priority,
                               cotenant_alone=cot_alone, cotenant_ms=cot_ms, **res)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if cot is not None:
                        print(f"[r{rank}]   co-tenant {res['cotenant_rate']:.1f} batches/s "
                              f"({100 * res['cotenant_rate'] / cot_alone:.0f}% of alone)", flush=True)
                    print(f"[r{rank}] {label:9s} {a.config} L={a.L:5d} {f:4.2f}x = {q:7.1f}/s achieved "
                          f"{rec['achieved']:7.1f}/s batch {rec['mean_batch']:4.1f} p50 {rec['p50_ms']:7.2f} p99 "
                          f"{rec['p99_ms']:7.2f} ms; within SLO {100 * rec['success']:5.1f}%; GPU {100 * rec['gpu_share']:4.1f}%;"
                          f" cores {rec['cores_other']:.2f}; checks {rec.get('checks', 0)} bad {rec.get('bad_words', 0)}",
                          flush=True)
                    if a.stop == "sat":
                        fail = rec["achieved"] < 0.97 * q or rec["p99_ms"] > a.sat_p99_ms
                    else:
                        fail = rec["p99_ms"] > a.slo_ms or rec["success"] < 0.99 or rec["achieved"] < 0.9 * q   # Poisson: +-5% at 5 s
                    bad = bad + 1 if fail else 0
                    if bad >= 2:
                        break
                srv.close()
                if cot is not None:
                    cot.close()
        out.close()
    if a.strategy == "fetch":
        hd.barrier()
        tr.destroy()


if __name__ == "__main__":
    main()
