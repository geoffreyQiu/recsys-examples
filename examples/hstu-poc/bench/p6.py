"""P6: GPU-SM vs CPU-core balance of the transfer work (study doc §10, §12).

(a) SM contention. A hog kernel fills every SM slot for --hog-ms on another stream; 2 ms later one request's KV
    transfer is issued, and we time issue -> "data usable" (GIN/raw: a wait kernel has seen the last layer complete;
    CPU verbs: the worker has seen it). Variants:
      none       no hog (baseline)
      hog        hog on a normal stream, transfer kernels on a normal stream
      hog-prio   transfer kernels on a high-priority stream (priority does not preempt resident blocks)
      hog-green  hog in a green context without N SMs; transfer kernels in a green context with those N SMs
(b) CPU contention (CPU verbs): a busy process on the worker thread's core while transferring.
Runs on 2 nodes like p2.py:  scripts/srun2.sh -j J -n 1 -- python bench/p6.py --transport gin|raw|verbs
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2 import NICS, WORKER_CORES, fill_pool  # noqa: E402

from hstu_poc import dist as hd  # noqa: E402
from hstu_poc.ext import ext  # noqa: E402
from hstu_poc.kvsource import GinKVSource, RawGdakiSource, VerbsKVSource  # noqa: E402
from hstu_poc.model import CONFIGS, Batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True, choices=["gin", "raw", "verbs"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--config", default="M")
    ap.add_argument("--cells", default="1x1024,2x4096", help="BxL list")
    ap.add_argument("--hog-ms", type=float, default=20.0)
    ap.add_argument("--green-sms", type=int, default=2)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--pool-gb", type=float, default=16)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank, world, local = hd.init()
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    E = ext()
    label = a.label or a.transport
    pool = int(a.pool_gb) * 2**30
    if a.transport == "gin":
        uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
        tr = E.GinKV(uid, rank, world, pool, not infer, n_ctx=6, queue_depth=2048, strong_signals=True, n_signals=4)
        make = lambda cfg: GinKVSource(cfg, tr, E, peer, pool, n_ctx=4)  # noqa: E731
    elif a.transport == "raw":
        tr = E.RawGdaki(NICS[local], pool, 4, 4096) if infer else E.VerbsKV(NICS[local], pool, False, 4, 64, 0, -1)
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: RawGdakiSource(cfg, tr, E, pool)  # noqa: E731
    else:
        tr = E.VerbsKV(NICS[local], pool, infer, 4, 64, 0, WORKER_CORES[local])
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: VerbsKVSource(cfg, tr, E, pool)  # noqa: E731
    if not infer:
        fill_pool(E, tr.local(), a.seed)
    hd.barrier()
    if infer:
        cfg = CONFIGS[a.config]
        n_sm = torch.cuda.get_device_properties(0).multi_processor_count
        g_small, g_rest, n_small, n_rest = E.green_streams(a.green_sms)
        s_small, s_rest = torch.cuda.ExternalStream(g_small), torch.cuda.ExternalStream(g_rest)
        s_hog = torch.cuda.Stream()
        s_norm = torch.cuda.Stream()
        s_prio = torch.cuda.Stream(priority=-5)
        out = open(f"{a.out}.rank{rank}.jsonl", "a")
        hog_ns = int(a.hog_ms * 1e6)
        variants = ["none", "hog", "hog-prio", "hog-green"] + (["cpu-busy"] if a.transport == "verbs" else [])
        for cell in a.cells.split(","):
            B, L = map(int, cell.split("x"))
            b = Batch([L] * B, [256] * B)
            src = make(cfg)
            src.post_outside = True
            src.setup(b)
            for var in variants:
                ts = []
                for t in range(a.trials):
                    torch.cuda.synchronize()
                    hogp = None
                    if var.startswith("hog"):
                        with torch.cuda.stream(s_rest if var == "hog-green" else s_hog):
                            E.hog(hog_ns, n_sm * 16, 128)
                        time.sleep(0.002)                 # the hog is resident on every SM slot
                    if var == "cpu-busy":
                        hogp = subprocess.Popen(["taskset", "-c", str(WORKER_CORES[local]), sys.executable, "-c",
                                                 "import time\nt=time.time()\nwhile time.time()-t<30: pass"])
                        time.sleep(0.2)
                    st = {"none": s_norm, "hog": s_norm, "hog-prio": s_prio, "hog-green": s_small,
                          "cpu-busy": s_norm}[var]
                    t0 = time.perf_counter()
                    if isinstance(src, VerbsKVSource):
                        src.seq += 1
                        src.v.post_reads(src.tables[0], src.layer_ptr_cpu, src.seq)
                        src.v.wait_idle()
                    else:
                        with torch.cuda.stream(st):
                            src.post()
                            src.before_attn(cfg.layers - 1)   # completes when the last layer has landed
                        st.synchronize()
                    ts.append((time.perf_counter() - t0) * 1e3)
                    if hogp is not None:
                        hogp.kill()
                        hogp.wait()
                    torch.cuda.synchronize()
                rec = dict(transport=label, config=a.config, B=B, L=L, variant=var, hog_ms=a.hog_ms,
                           green_sms=n_small, ms_p50=float(np.median(ts)), ms_min=float(min(ts)), ms_max=float(max(ts)),
                           bytes=src.bytes_moved())
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"[r{rank}] {label:6s} B={B} L={L:5d} {var:9s}: issue->usable p50 {rec['ms_p50']:7.2f} ms "
                      f"(min {rec['ms_min']:.2f}, max {rec['ms_max']:.2f}); hog {a.hog_ms} ms; green {n_small}/{n_rest} SMs",
                      flush=True)
            src.release()
        out.close()
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
