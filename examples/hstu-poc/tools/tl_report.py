#!/usr/bin/env python3
"""Per-request timelines (results/tl*/<mode>.timeline.rank0.jsonl, bench/grid.py --timeline) against the SoL model
(tools/sol.py): where a request's time goes, and how far each piece is from its speed of light.

    tools/tl_report.py [results/tl] [--json OUT]

Per (mode, config, B, L), medians over the traced requests (times from the request's issue on the host):
  posted    the CPU finished posting (CPU verbs inline posting; GDAKI: the zero-copy descriptor store)
  launched  the graph launch call returned
  g0        the graph's first kernel ran (GPU)
  first     the first chunk step of layer 0 had landed on every queue pair (CPU verbs: the worker saw it; GDAKI: the
            post kernel published it)
  last      the last layer had fully landed (same observers)
  a0        layer 0's attention started (earliest CTA)
  aN        the last layer's attention ended (latest CTA)
  g1        the graph's last kernel ran
  seen      the host saw the output event complete (= the measured latency)
Gap analysis (fetch modes):
  start gap  first - SoL first (doorbell + one read round trip + the first step's bytes at line rate)
  rate       bytes / (last - first + first step's line-rate time): achieved NIC rate
  tail       seen - last: what follows the last byte; SoL tail = out GEMM + head + notification
  compute    local-hit latency (same cell) - SoL compute: the non-RDMA part, for compute-bound requests
"""
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sol  # noqa: E402

FULL = 0x7fffffff


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load(d):
    cells = collections.defaultdict(list)
    meas = {}
    for f in glob.glob(os.path.join(d, "*.timeline.rank0.jsonl")):
        for x in open(f):
            r = json.loads(x)
            k = (r["mode"], r["config"], int(r["B"]), int(r["L"]))
            cells[k].append(r)
            meas[k] = r.get("host_mean_ms")
    return cells, meas


def events(r):
    """[(layer, value, t)] of the transfer's progress: CPU verbs worker (host clock) or the GPU publisher."""
    if "cpu" in r and r["cpu"]["ev"]:
        return r["cpu"]["ev"]
    return [e for e in r.get("pub_ev", []) if e[2] is not None]


def summarize(rs, cfg):
    n = sol.CFG[cfg][0]
    out = {}

    def take(name, f):
        vals = []
        for r in rs:
            try:
                vals.append(f(r))
            except (KeyError, IndexError, TypeError, ValueError):
                vals.append(None)
        out[name] = med(vals)
    take("posted", lambda r: r["host"][1])
    take("launched", lambda r: r["host"][2])
    take("seen", lambda r: r["host"][3])
    take("g0", lambda r: r["g0"])
    take("g1", lambda r: r["g1"])
    take("a0", lambda r: r["attn"][0][0])
    take("aN", lambda r: r["attn"][n - 1][1])
    take("first", lambda r: min(t for l, v, t in events(r) if l == 0))
    take("last", lambda r: max(t for l, v, t in events(r)))
    take("pub_last", lambda r: max(t for l, v, t in r["pub_ev"] if t is not None))     # GPU-visible (relay/post)
    take("cpu_post", lambda r: r["cpu"]["post1"] - r["cpu"]["post0"])
    take("cqe0", lambda r: r["cpu"]["cqe0"])
    # per layer: attention start, end, last data wait end; data complete (observer) per layer
    lay = []
    for l in range(n):
        d = {}
        for nm, f in (("a_start", lambda r: r["attn"][l][0]), ("a_end", lambda r: r["attn"][l][1]),
                      ("a_wait", lambda r: r["attn"][l][2]),
                      ("data", lambda r: max(t for ll, v, t in events(r) if ll == l))):
            vals = []
            for r in rs:
                try:
                    vals.append(f(r))
                except (KeyError, IndexError, TypeError, ValueError):
                    vals.append(None)
            d[nm] = med(vals)
        lay.append(d)
    out["layers"] = lay
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tl")
    cells, meas = load(d)
    local = {(c, B, L): s for (m, c, B, L), s in meas.items() if m == "local"}
    res = []
    us = lambda x: "–" if x is None else f"{x / 1e3:.1f}"  # noqa: E731
    by_mode = collections.defaultdict(list)
    for k in sorted(cells, key=lambda k: (k[0], "SML".index(k[1]), k[2], k[3])):
        by_mode[k[0]].append(k)
    for m, ks in by_mode.items():
        fetch = m not in ("recompute", "local")
        print(f"\n### {m}\n")
        if fetch:
            print("| cell | latency | SoL | posted | g0 | first (SoL) | rate GB/s | last | a(N-1) end | g1 | seen | "
                  "tail (SoL) | local |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        else:
            print("| cell | latency | SoL | launched | g0 | a0 | a(N-1) end | g1 | seen |")
            print("|---|---|---|---|---|---|---|---|---|")
        for k in ks:
            _, c, B, L = k
            s = summarize(cells[k], c)
            n, h, H, dd = sol.CFG[c]
            if fetch:
                sl, parts = sol.sol_fetch(c, B, L)
                rows = cells[k][0].get("tile_rows") or L
                step_bytes = B * 2 * min(rows, L) * H * dd * 2
                first_sol = sol.LAT["t_rdma"] + step_bytes / sol.HW["nic"]
                tot = sol.kv_bytes(c, B, L)
                rate = None
                if s["first"] is not None and s["last"] is not None:
                    rate = tot / ((s["last"] - s["first"]) / 1e9 + step_bytes / sol.HW["nic"]) / 1e9
                ops = sol.ops_fetch(c, B, L)
                tail_sol = ops[-2]["t"] + ops[-1]["t"] + sol.LAT["t_out"]
                lo = local.get((c, B, L))
                print(f"| {c} B={B} {L / 1024:g}k | {meas[k]:.3f} | {sl * 1e3:.3f} | {us(s['posted'])} | {us(s['g0'])} | "
                      f"{us(s['first'])} ({first_sol * 1e6:.1f}) | {rate and f'{rate:.1f}' or '–'} | {us(s['last'])} | "
                      f"{us(s['aN'])} | {us(s['g1'])} | {us(s['seen'])} | "
                      f"{us(None if s['last'] is None else s['seen'] - s['last'])} ({tail_sol * 1e6:.1f}) | "
                      f"{'–' if lo is None else f'{lo:.3f}'} |")
                res.append(dict(mode=m, config=c, B=B, L=L, latency_ms=meas[k], sol_ms=sl * 1e3,
                                first_sol_us=first_sol * 1e6, tail_sol_us=tail_sol * 1e6, rate_gbs=rate,
                                local_ms=lo, transfer_sol_ms=parts["transfer"] * 1e3,
                                compute_sol_ms=parts["compute"] * 1e3, **s))
            else:
                sl = sol.sol_recompute(c, B, L) if m == "recompute" else sol.sol_fetch(c, B, L)[1]["compute"]
                print(f"| {c} B={B} {L / 1024:g}k | {meas[k]:.3f} | {sl * 1e3:.3f} | {us(s['launched'])} | "
                      f"{us(s['g0'])} | {us(s['a0'])} | {us(s['aN'])} | {us(s['g1'])} | {us(s['seen'])} |")
                res.append(dict(mode=m, config=c, B=B, L=L, latency_ms=meas[k], sol_ms=sl * 1e3, **s))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
