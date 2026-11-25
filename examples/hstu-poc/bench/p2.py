"""P2 driver: workload R over remote transports on 2 nodes (scripts/srun2.sh -n N -- python bench/p2.py ...).
Ranks [0, N) on node A run the grid cells (bench/grid.py) with --transport as the KV source; ranks [N, 2N) on
node B hold the store pool, filled with a seeded pattern by byte offset, so every fetch can be checked.
  gin    NCCL GIN, backend by NCCL_GIN_TYPE (3 = GDAKI, 2 = Proxy); store pool = cuMemCreate HOST_NUMA window
  verbs  CPU verbs; store pool = mmap'd host memory bound to the NIC's NUMA node, never touched by CUDA
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid  # noqa: E402

from hstu_poc import dist as hd  # noqa: E402
from hstu_poc.ext import ext  # noqa: E402
from hstu_poc.kvsource import GinKVSource, RawGdakiSource, VerbsKVSource  # noqa: E402

NICS = ["mlx5_0", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_9", "mlx5_10", "mlx5_11"]
WORKER_CORES = [48, 50, 52, 54, 104, 106, 108, 110]   # on the socket of GPU/NIC k (DGX H100)


def fill_pool(E, buf, seed, chunk=1 << 30):
    for off in range(0, buf.numel(), chunk):
        n = min(chunk, buf.numel() - off) // 8 * 8
        E.fill_pattern_host(buf[off:off + n], 0, 0, off // 8, seed)


def main():
    ap = argparse.ArgumentParser()
    grid.add_args(ap)
    ap.add_argument("--transport", required=True, choices=["gin", "verbs", "raw"])
    ap.add_argument("--pool-gb", type=float, default=48)
    ap.add_argument("--local-gb", type=float, default=36)
    ap.add_argument("--nctx", type=int, default=4, help="GIN contexts for reads (P2: 4 needed for line rate with queued reads)")
    ap.add_argument("--nqp", type=int, default=2, help="verbs QPs")
    ap.add_argument("--flush", type=int, default=0, help="verbs: 1 = cuFlushGPUDirectRDMAWrites per layer")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--variants", type=int, default=8, help="user sets rotated over requests (distinct users)")
    ap.add_argument("--label", default=None, help="mode name of the transport in --modes and the results (default: --transport)")
    a = ap.parse_args()
    rank, world, local = hd.init()
    from hstu_poc.kvsource import RemoteKV
    RemoteKV.variants = a.variants
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    infer_group = hd.subgroup(range(n))
    E = ext()
    a.ext = E
    pool = int(a.pool_gb * 2**30)
    t0 = time.time()
    if a.transport == "gin":
        uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
        # NCCL 2.32.7 ("elastic GIN") needs identical window sizes on all ranks: local window = store pool size
        nbytes = int(max(a.local_gb, a.pool_gb)) * 2**30
        pool = nbytes
        tr = E.GinKV(uid, rank, world, nbytes, not infer, n_ctx=a.nctx + 2, queue_depth=2048, strong_signals=True,
                     n_signals=4)
        make = lambda cfg: GinKVSource(cfg, tr, E, peer, pool, n_ctx=a.nctx)  # noqa: E731
    elif a.transport == "raw":
        # raw GDAKI on the inference side; the store is the same GPU-less verbs pool as for CPU verbs
        if infer:
            tr = E.RawGdaki(NICS[local], int(a.local_gb * 2**30), a.nqp, 4096)
        else:
            tr = E.VerbsKV(NICS[local], pool, False, a.nqp, 64, 0, -1)
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: RawGdakiSource(cfg, tr, pool)  # noqa: E731
    else:
        nbytes = int(a.local_gb * 2**30) if infer else pool
        tr = E.VerbsKV(NICS[local], nbytes, infer, a.nqp, 64, a.flush, WORKER_CORES[local])
        infos = hd.gather(tr.local_info())
        tr.connect(infos[peer])
        make = lambda cfg: VerbsKVSource(cfg, tr, E, pool)  # noqa: E731
    if not infer:
        fill_pool(E, tr.local(), a.seed)
    hd.barrier()
    if rank == 0:
        extra = f" GDR writes ordering attr {tr.gdr_writes_ordering()}" if a.transport == "verbs" else ""
        print(f"P2 {a.transport} (NCCL_GIN_TYPE={os.environ.get('NCCL_GIN_TYPE', '-')}) ready in "
              f"{time.time() - t0:.0f} s: {n} inference + {n} store ranks, pool {a.pool_gb} GB/rank{extra}", flush=True)
    if infer:
        hd.GROUP = infer_group
        grid.run_grid(a, rank, n, {a.label or a.transport: make, "local_buf": tr.local()}, gpus=n)
        hd.GROUP = None
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
