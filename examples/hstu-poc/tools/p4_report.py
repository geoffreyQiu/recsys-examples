#!/usr/bin/env python3
"""P4b report (wait variants on a regular grid) from results/p4b/: markdown tables.
    tools/p4_report.py results/p4b
Variants: 5a (wait kernel before each layer's attention, CUDA graph), 5a eager (no graph), pdl (attention launched
with programmatic dependent launch after the wait kernel), fused (attention's producer warp waits by itself)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

NAMES = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "verbs": "CPU verbs", "proxy": "GIN-Proxy"}
VARS = ["5a", "5a_eager", "pdl", "fused"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    lat = defaultdict(dict)       # (transport, config, L, B) -> variant -> mean ms
    for f in glob.glob(f"{a.dir}/*.jsonl"):
        t, v = os.path.basename(f).split(".")[0].split("_", 1)
        for x in open(f):
            if not x.strip():
                continue
            r = json.loads(x)
            if "skipped" in r:
                continue
            lat[(t, r["config"], int(r["L"]), int(r["B"]))][v] = r["mean_ms"]
    trs = [t for t in NAMES if any(k[0] == t for k in lat)]
    print("**Change of mean request latency vs 5′a in a CUDA graph (%), over the grid "
          "(S and M × L 1k–16k × B 1–8): median [min, max]**\n")
    print("| Transport | " + " | ".join(v.replace("_", " ") for v in VARS[1:]) + " |")
    print("|---|" + "---|" * (len(VARS) - 1))
    for t in trs:
        cells = []
        for v in VARS[1:]:
            d = [100 * (m[v] / m["5a"] - 1) for k, m in lat.items() if k[0] == t and v in m and "5a" in m]
            cells.append(f"{np.median(d):+.1f} [{min(d):+.1f}, {max(d):+.1f}]" if d else "–")
        print(f"| {NAMES[t]} | " + " | ".join(cells) + " |")
    for c in ("S", "M"):
        print(f"\n**Config {c}: mean latency (ms) with 5′a in a CUDA graph; in brackets the change with PDL / fused (%)**\n")
        Bs = sorted({k[3] for k in lat if k[1] == c})
        Ls = sorted({k[2] for k in lat if k[1] == c})
        for t in [x for x in ("raw", "gdaki") if x in trs]:
            print(f"\n{NAMES[t]}:\n")
            print("| L \\ B | " + " | ".join(str(B) for B in Bs) + " |")
            print("|---|" + "---|" * len(Bs))
            for L in Ls:
                row = []
                for B in Bs:
                    m = lat.get((t, c, L, B), {})
                    if "5a" not in m:
                        row.append("–")
                        continue
                    d = lambda v: f"{100 * (m[v] / m['5a'] - 1):+.1f}" if v in m else "–"  # noqa: E731
                    row.append(f"{m['5a']:.3g} ({d('pdl')} / {d('fused')})")
                print(f"| {L // 1024}k | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
