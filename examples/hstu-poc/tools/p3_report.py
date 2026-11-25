#!/usr/bin/env python3
"""P3b (workload W, regular grid) report: tables and a figure from results/p3b/.
    tools/p3_report.py results/p3b -o docs/figs/p3b-tail.png [--md]
Per cell (config M, C = 256): scores = time until the candidate scores exist (the request's answer), with and without
the writes; durable = time until the new tokens' K/V are confirmed in the remote store; tail = durable - scores."""
import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential blue, ordinal (P)
NAMES = {"gdaki": "GIN-GDAKI", "proxy": "GIN-Proxy", "verbs": "CPU verbs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(f"{a.dir}/*.jsonl"):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    rows = [r for r in rows if "skipped" not in r]
    cell = defaultdict(dict)     # (B, P, S) -> mode -> row
    for r in rows:
        cell[(r["B"], r["P"], r["S"])][r["mode"]] = r
    Bs = sorted({k[0] for k in cell})
    Ps = sorted({k[1] for k in cell})
    Ss = sorted({k[2] for k in cell})
    modes = [m for m in ("gdaki", "verbs", "proxy") if any(m in v for v in cell.values())]
    fig, axes = plt.subplots(1, len(Bs), figsize=(3.4 * len(Bs), 3.2), dpi=140, facecolor=SURFACE, squeeze=False,
                             sharey=True)
    for ax, B in zip(axes[0], Bs):
        ax.set_facecolor(SURFACE)
        for i, P in enumerate(Ps):
            xs, ys = [], []
            for S in Ss:
                m = cell.get((B, P, S), {})
                if "gdaki" in m:
                    xs.append(S)
                    ys.append(max(m["gdaki"]["tail_mean"], 0.0))
            if xs:
                ax.plot(xs, ys, color=RAMP[i % len(RAMP)], lw=1.8, marker="o", ms=4.5, mec=SURFACE, mew=1,
                        label=f"prefix P = {P // 1024}k" if P else "P = 0 (fresh)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ss)
        ax.set_xticklabels([f"{S / 1024:g}k" if S >= 1024 else str(S) for S in Ss], fontsize=7.5, color=INK2)
        ax.set_title(f"B = {B}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("new tokens written per user (S)", color=INK2, fontsize=8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("durable − scores ready (ms)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=len(Ps), frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.08))
    fig.suptitle("Workload W, M, C = 256, GIN-GDAKI writes: how long the new K/V stay in flight after the scores are ready",
                 color=INK, fontsize=10, y=1.15, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if not a.md:
        return
    for B in Bs:
        print(f"\n**B = {B}: scores ready without writes → with GIN-GDAKI writes / durable (mean ms)**\n")
        print("| P \\ S | " + " | ".join(f"{S / 1024:g}k" if S >= 1024 else str(S) for S in Ss) + " |")
        print("|---|" + "---|" * len(Ss))
        for P in Ps:
            cells = []
            for S in Ss:
                m = cell.get((B, P, S), {})
                if "none" in m and "gdaki" in m:
                    cells.append(f"{m['none']['scores_mean']:.2f} → {m['gdaki']['scores_mean']:.2f} / "
                                 f"{m['gdaki']['durable_mean']:.2f}")
                else:
                    cells.append("–")
            print(f"| {P // 1024}k | " + " | ".join(cells) + " |")
    print("\n**Durable (mean ms) by transport, and CPU ms per request, at selected cells**\n")
    print("| B | P | S | written | " + " | ".join(f"durable {NAMES[m]}" for m in modes) + " | "
          + " | ".join(f"CPU {NAMES[m]}" for m in modes) + " |")
    print("|---|---|---|---|" + "---|" * (2 * len(modes)))
    for B in Bs:
        for P in (Ps[0], Ps[-1]):
            for S in (Ss[0], Ss[-1]):
                m = cell.get((B, P, S), {})
                if not all(x in m for x in modes):
                    continue
                w = m[modes[0]]["bytes"] / 2**20
                print(f"| {B} | {P // 1024}k | {S} | {w:,.0f} MiB | "
                      + " | ".join(f"{m[x]['durable_mean']:.2f}" for x in modes) + " | "
                      + " | ".join(f"{m[x]['cpu_ms_per_req']:.2f}" for x in modes) + " |")
    worst = max((abs(c[m]["scores_mean"] / c["none"]["scores_mean"] - 1), k, m) for k, c in cell.items()
                for m in modes if m in c and "none" in c)
    bad = sum(c[m].get("readback_bad_bytes", 0) for c in cell.values() for m in modes if m in c)
    print(f"\nLargest change of scores-ready time from writes: {100 * worst[0]:.1f}% at (B, P, S) = {worst[1]} "
          f"({worst[2]}); read-back mismatches over all cells: {bad} bytes")


if __name__ == "__main__":
    main()
