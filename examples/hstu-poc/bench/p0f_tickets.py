"""P0-f: GIN split-phase across kernels (study doc §5.1), with data checks.

Read: kernel A posts every layer's gets and stores one ticket per layer, then exits; later kernels wait on
ticket l and check layer l's data against the store's seeded pattern.
  immediate   wait(l) + check(l) right after the post kernel, layer by layer
  busy        a 5 ms spin kernel between post and the waits
  interleaved wait(l), check(l), 200 us spin (a "layer"), next l
  no-wait     negative control: check the last layer right after posting, without waiting (expect mismatches)
Write: fill a local region, put it (kernel A, ticket), wait the ticket (kernel B), commit with a strong signal
(kernel C); the store waits for the signal on its GPU and checks its host memory from the CPU.

Run with 2N ranks (N inference ranks on node A, N store ranks on node B); NCCL_GIN_TYPE picks the backend.
"""
import argparse
import os

import torch

from hstu_poc import dist as hd
from hstu_poc.ext import ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=4096, help="cached tokens per user")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tok-bytes", type=int, default=4096, help="K+V bytes per token per layer (M config: 4096)")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--nctx", type=int, default=2, help="GIN contexts (QPs) the reads are spread over")
    a = ap.parse_args()

    rank, world, local = hd.init()
    n = world // 2
    reader = rank < n
    peer = rank + n if reader else rank - n
    E = ext()
    uid = hd.bcast(E.gin_unique_id() if rank == 0 else None)
    L, U, T, ML, TB = a.layers, a.users, a.tokens, a.max_len, a.tok_bytes
    slot = L * ML * TB                                  # store bytes per user
    lstride = U * T * TB                                # local bytes per layer
    # NCCL 2.32.7 ("elastic GIN") requires the same window segment size on every rank: register max of both, 1 GiB-rounded
    nbytes = max(U * slot, L * lstride + U * T * TB)   # reader: L layers + a write-source region
    nbytes = (nbytes + (1 << 30) - 1) >> 30 << 30
    g = E.GinKV(uid, rank, world, nbytes, not reader, n_ctx=a.nctx + 2, queue_depth=1024, strong_signals=True, n_signals=4)
    buf = g.local()
    backend = os.environ.get("NCCL_GIN_TYPE", "?")
    dev = torch.device("cuda", local)

    segs, lp, rows = [], [0], []
    for l in range(L):
        for u in range(U):
            remote, loc = u * slot + l * ML * TB, l * lstride + u * T * TB
            segs.append([remote, loc, T * TB, u])
            rows.append([loc, T * TB, u, l << 40])
        lp.append(len(segs))
    segs_t = torch.tensor(segs, dtype=torch.int64, device=dev)
    lp_t = torch.tensor(lp, dtype=torch.int32, device=dev)
    tickets = torch.zeros(L * a.nctx, 2, dtype=torch.int64, device=dev)
    rows_l = [torch.tensor(rows[l * U:(l + 1) * U], dtype=torch.int64, device=dev) for l in range(L)]
    bad = torch.zeros(L, dtype=torch.int64, device=dev)
    t_post = torch.zeros(2, dtype=torch.int64, device=dev)
    t_done = torch.zeros(L, dtype=torch.int64, device=dev)
    # write test: user 0's layer 0 region of the store slot (tokens [T, 2T) so it never overlaps the read data)
    wsrc = L * lstride
    wrow_local = torch.tensor([[wsrc, T * TB, 7, (3 << 40) | 0]], dtype=torch.int64, device=dev)
    wseg = torch.tensor([[T * TB, wsrc, T * TB, 0]], dtype=torch.int64, device=dev)
    wrow_store = torch.tensor([[T * TB, T * TB, 7, (3 << 40) | 0]], dtype=torch.int64)
    wticket = torch.zeros(2, dtype=torch.int64, device=dev)

    results = []
    variants = ["immediate", "busy", "interleaved", "no-wait"]
    for it in range(a.iters):
        for vi, var in enumerate(variants):
            seed = 1000 + it * 10 + vi
            if not reader:
                for u in range(U):
                    for l in range(L):
                        off = u * slot + l * ML * TB
                        E.fill_pattern_host(buf[off:off + T * TB], u, l, 0, seed)
            hd.barrier()
            if reader:
                buf[:L * lstride].zero_()
                bad.zero_(); t_done.zero_()
                torch.cuda.synchronize()
                g.post_gets(segs_t, lp_t, tickets, peer, ctx0=0, n_ctx=a.nctx, t_post=t_post)
                if var == "no-wait":
                    E.check_pattern_gpu(buf, rows_l[L - 1], seed, bad[L - 1:L])
                    for l in range(L):
                        g.wait_layer(tickets, l, 0, a.nctx, t_done)
                else:
                    if var == "busy":
                        E.spin(5_000_000)
                    for l in range(L):
                        g.wait_layer(tickets, l, 0, a.nctx, t_done)
                        E.check_pattern_gpu(buf, rows_l[l], seed, bad[l:l + 1])
                        if var == "interleaved":
                            E.spin(200_000)
                torch.cuda.synchronize()
                tp, td, b = t_post.cpu().tolist(), t_done.cpu().tolist(), bad.cpu().tolist()
                done_us = [(x - tp[0]) / 1e3 for x in td]
                gbs = L * U * T * TB / max(td[-1] - tp[0], 1)
                results.append((var, it, sum(b) if var != "no-wait" else b[L - 1], (tp[1] - tp[0]) / 1e3, done_us, gbs))
            hd.barrier()
        # write + commit
        wseed = 5000 + it
        if reader:
            E.fill_pattern_gpu(buf, wrow_local, wseed)
            g.put_segs(wseg, wticket, peer, ctx=a.nctx, threads=1)
            g.wait_ticket(wticket, ctx=a.nctx)
            g.commit(peer, ctx=a.nctx, sig=0)
            torch.cuda.synchronize()
        else:
            g.wait_signal(ctx=a.nctx, sig=0, value=it + 1)
            torch.cuda.synchronize()
            wbad = E.check_pattern_host(buf, wrow_store, wseed)
            sigv = g.read_signal(a.nctx, 0)
            results.append(("write+commit", it, wbad, sigv))
        hd.barrier()

    allres = hd.gather(results)
    if rank == 0:
        print(f"P0-f backend NCCL_GIN_TYPE={backend}: {L} layers x {U} users x {T} tokens x {TB} B "
              f"({L * U * T * TB / 2**20:.0f} MiB per request), world {world}")
        for r, res in enumerate(allres):
            for x in res:
                if x[0] == "write+commit":
                    print(f"  rank {r} {x[0]:12s} it {x[1]}: store-side mismatched words {x[2]} (signal value {x[3]})")
                else:
                    var, it, nb, post_us, done_us, gbs = x
                    ok = (nb > 0) if var == "no-wait" else (nb == 0)
                    print(f"  rank {r} {var:12s} it {it}: {'PASS' if ok else 'FAIL'} mismatches {nb}; post kernel "
                          f"{post_us:.1f} us; layer done at " + " ".join(f"{d:.0f}" for d in done_us) +
                          f" us; {gbs:.2f} GB/s")
    hd.barrier()
    g.destroy()


if __name__ == "__main__":
    main()
