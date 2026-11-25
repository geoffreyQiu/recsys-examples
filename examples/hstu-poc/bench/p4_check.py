"""P4 correctness under a slow producer (study doc §5.4): each wait variant must produce the same output as
the 5'a reference even when posting is delayed and the local KV rows are zeroed before the request (a variant that
does not really wait computes on zeros). Negative control: 'nowait' (copied kernel, no wait at all) must differ.
Runs on 2 nodes like p2.py:  scripts/srun2.sh -j J -n 1 -- python bench/p4_check.py --transport raw|gin|verbs"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2 import NICS, WORKER_CORES, fill_pool  # noqa: E402

from hstu_poc import dist as hd  # noqa: E402
from hstu_poc.ext import ext  # noqa: E402
from hstu_poc.kvsource import GinKVSource, RawGdakiSource, RemoteKV, VerbsKVSource  # noqa: E402
from hstu_poc.model import CONFIGS, HSTU, Batch  # noqa: E402


class NoWait:
    """Negative control: copied kernel, no wait kernel, no in-kernel wait."""
    def __init__(self, src):
        self.src = src

    def __call__(self, l, q, K, V, b, model):
        from hstu_poc.attn_ext import hstu_fwd
        return hstu_fwd(q, K, V, b.cu_q, b.cu_k, b.max_q, b.max_k, model.cfg.max_len, b.num_targets, model.alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True, choices=["gin", "verbs", "raw"])
    ap.add_argument("--config", default="M")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--slow-ms", type=float, default=2.0)
    ap.add_argument("--pool-gb", type=float, default=16)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    rank, world, local = hd.init()
    n = world // 2
    infer = rank < n
    peer = rank + n if infer else rank - n
    E = ext()
    RemoteKV.variants = 1                     # the same users for every request: identical data across variants
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
        model = HSTU(cfg)
        b = Batch([a.L] * a.B, [a.C] * a.B)
        items = torch.randint(0, cfg.n_items, (b.total_q,), device="cuda", generator=torch.Generator("cuda").manual_seed(5))
        out = {}
        with torch.inference_mode():
            for variant in ["5a", "copy", "pdl", "fused", "pdl_early", "nowait"]:
                slow = 0 if variant == "5a" else int(a.slow_ms * 1e6)
                src = make(cfg).p4_init("copy" if variant == "nowait" else variant, slow_ns=slow)
                if variant == "nowait":
                    src.attn_fn = NoWait(src)
                src.setup(b)
                verbs = isinstance(src, VerbsKVSource)
                res = {}

                def fwd():
                    src.start()
                    x = model.embed(items, None, b.pos_q)
                    res["y"] = model.forward_cached(x, b, src.kv, before_attn=src.before_attn, attn_fn=src.attn_fn)
                    src.end()

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):      # warm-up (also builds the copied kernel on first use)
                    src.pre_replay(0)
                    fwd()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fwd()
                for trial in range(3):
                    torch.cuda.synchronize()
                    src.buf.zero_()                 # history rows must come from this request's transfer
                    torch.cuda.synchronize()
                    if verbs and slow:              # slow producer on the CPU side: launch first, post later
                        src.post_after_launch = True
                        src.pre_replay(0)
                        g.replay()
                        time.sleep(a.slow_ms / 1e3)
                        src.post_now()
                        src.post_after_launch = False
                    else:
                        src.pre_replay(0)
                        g.replay()
                    torch.cuda.synchronize()
                    out.setdefault(variant, []).append(res["y"].float().clone())
                del g
                src.release()
        ref = out["5a"][0]
        print(f"P4 slow-producer check, {a.transport}, {a.config} B={a.B} L={a.L} C={a.C}, posting delayed "
              f"{a.slow_ms} ms, KV rows zeroed before each request:")
        for v, ys in out.items():
            diffs = [float((y - ref).abs().max()) for y in ys]
            expect_equal = v != "nowait"
            ok = all(d == 0.0 for d in diffs) if expect_equal else all(d > 0.0 for d in diffs)
            print(f"  {v:7s} max |output - 5a| per trial: {', '.join(f'{d:.3g}' for d in diffs)}  ->  "
                  f"{'PASS' if ok else 'FAIL'} ({'must equal' if expect_equal else 'must differ'})", flush=True)
    hd.barrier()
    tr.destroy()


if __name__ == "__main__":
    main()
