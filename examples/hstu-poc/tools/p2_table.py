#!/usr/bin/env python3
"""Markdown tables from the P2c grid: mean request latency per transport (+ local hit, recompute), and the small-request
ranking.   tools/p2_table.py results/p2c [--config M] [--B 1,2,4,8] [--L 1024,2048,4096,8192,16384]"""
import argparse
import glob
import json

T = ["raw", "gdaki", "verbs", "proxy"]
NAMES = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "verbs": "CPU verbs", "proxy": "GIN-Proxy", "local": "local hit",
         "recompute": "recompute"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--config", default="M")
    ap.add_argument("--B", default="1,2,4,8")
    ap.add_argument("--L", default="1024,2048,4096,8192,16384")
    a = ap.parse_args()
    lat = {}
    for f in glob.glob(f"{a.dir}/*.jsonl"):
        for x in open(f):
            r = json.loads(x)
            if "skipped" not in r and str(r["L"]) != "skew":
                lat[(r["mode"], r["config"], int(r["B"]), int(r["L"]))] = r["mean_ms"]
    c = a.config
    print("| L | B | " + " | ".join(NAMES[t] for t in T) + " | local hit | recompute |")
    print("|---|---|" + "---|" * (len(T) + 2))
    for L in map(int, a.L.split(",")):
        for B in map(int, a.B.split(",")):
            v = [lat.get((t, c, B, L)) for t in T]
            best = min(x for x in v if x is not None)
            cells = [(f"**{x:.3g}**" if x == best else f"{x:.3g}") if x is not None else "–" for x in v]
            cells += [f"{lat[(m, c, B, L)]:.3g}" if (m, c, B, L) in lat else "–" for m in ("local", "recompute")]
            print(f"| {L // 1024}k | {B} | " + " | ".join(cells) + " |")
    # small-request ranking over every config, B and L: relative to the fastest transport per cell
    rel = {t: [] for t in T}
    spread = []
    for key in {k[1:] for k in lat}:
        v = {t: lat[(t,) + key] for t in T if (t,) + key in lat}
        if len(v) < len(T):
            continue
        best = min(v.values())
        spread.append((max(v.values()) / best - 1, key, best))
        for t, x in v.items():
            rel[t].append((x / best - 1, key, best))
    big = [s for s in spread if s[0] > 0.01]
    print(f"\ncells with all four transports: {len(spread)}; spread > 1%: {len(big)}; "
          f"largest fastest-transport latency among those: {max((s[2] for s in big), default=0):.2f} ms")
    for t in T:
        xs = sorted(rel[t], reverse=True)
        print(f"{NAMES[t]:10s}: slower than the fastest by > 1% in {sum(1 for x in xs if x[0] > 0.01)} cells; "
              f"max +{100 * xs[0][0]:.1f}% at {xs[0][1]} (fastest {xs[0][2]:.3f} ms)")


if __name__ == "__main__":
    main()
