#!/usr/bin/env python3
"""Flatten the P2t grid (results/p2t, host clock) into the P2c row format used by report_figs.py and
p2c_breakeven.py: per cell, mode verbs|raw|gdaki|proxy = the fastest measured implementation of that transport (basic
or with latency options, any wait variant), plus recompute and local; mean_ms/p50_ms = host-clock mean/median.
    tools/p2t_to_p2c.py results/p2t work/p2t_flat"""
import glob
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
best = {}
for f in glob.glob(f"{src}/*.jsonl"):
    for x in open(f):
        r = json.loads(x)
        if "skipped" in r or "host_mean_ms" not in r or r["mode"].split(".")[1:2] == ["b2"]:
            continue
        mode = r["mode"].split(".")[0]
        k = (mode, r["config"], int(r["B"]), int(r["L"]))
        row = dict(r, mode=mode, mean_ms=r["host_mean_ms"], p50_ms=r["host_p50_ms"], p99_ms=r["host_p99_ms"],
                   impl=r["mode"])
        if k not in best or row["mean_ms"] < best[k]["mean_ms"]:
            best[k] = row
with open(f"{dst}/flat.rank0.jsonl", "w") as out:
    for row in best.values():
        out.write(json.dumps(row) + "\n")
print(f"wrote {len(best)} cells to {dst}/flat.rank0.jsonl")
