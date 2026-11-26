#!/usr/bin/env python3
"""Same-pair latency comparison of implementations measured in one results directory (host-clock means).
    tools/tl_compare.py results/tl2 verbs.t2.tile:verbs.t.tile raw.t2.tile:raw.t.tile ...
Prints, per pair A:B, the per-cell change of A vs B (negative = A faster) and a summary (faster / tie / slower beyond
the 0.95% run-to-run noise)."""
import glob
import json
import os
import statistics
import sys

NOISE = 0.0095


def load(d):
    m = {}
    for f in glob.glob(os.path.join(d, "*.rank0.jsonl")):
        if ".timeline." in f:
            continue
        for x in open(f):
            r = json.loads(x)
            if "host_mean_ms" in r:
                m.setdefault(r["mode"], {})[(r["config"], int(r["B"]), int(r["L"]))] = r["host_mean_ms"]
    return m


def main():
    m = load(sys.argv[1])
    for pair in sys.argv[2:]:
        a, b = pair.split(":")
        cells = sorted(set(m.get(a, {})) & set(m.get(b, {})), key=lambda c: ("SML".index(c[0]), c[1], c[2]))
        d = [(m[a][c] / m[b][c] - 1, c) for c in cells]
        if not d:
            continue
        fa = sum(x < -NOISE for x, _ in d)
        fb = sum(x > NOISE for x, _ in d)
        print(f"\n{a} vs {b}: median {100 * statistics.median(x for x, _ in d):+.2f}%, faster/tie/slower {fa}/{len(d) - fa - fb}/{fb}")
        print("  " + "  ".join(f"{c[0]}{c[1]}x{c[2] // 1024 if c[2] >= 1024 else c[2] / 1024:g}k:{100 * x:+.1f}%" for x, c in d))
        print("  ms: " + "  ".join(f"{c[0]}{c[1]}x{c[2] / 1024:g}k:{m[a][c]:.3f}/{m[b][c]:.3f}" for x, c in d if abs(x) > NOISE))


if __name__ == "__main__":
    main()
