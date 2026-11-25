#!/usr/bin/env python3
"""P5b at node scale: all 8 GPUs of the inference node serve at once, each with its own NIC and store rank.
    tools/p5_eight.py results/p5b -o docs/figs/p5-eight.png [--slo 100] [--md]
Per transport, history length and offered rate per GPU: node throughput (sum of the 8 achieved rates), the worst
GPU's p99, the lowest GPU's success share, and the CPU cores the transport threads used on the node."""
import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["raw", "gdaki", "verbs", "proxy"]          # fixed categorical slots 1-4
COLORS = {"raw": "#2a78d6", "gdaki": "#eb6834", "verbs": "#1baf7a", "proxy": "#eda100"}
MARK = {"raw": "o", "gdaki": "s", "verbs": "^", "proxy": "D"}
NAMES = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "verbs": "CPU verbs", "proxy": "GIN-Proxy"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--slo", type=float, default=100.0)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(f"{a.dir}/eight_*.jsonl"):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    pts = defaultdict(list)
    for r in rows:
        pts[(r["strategy"], r["L"], r["offered"])].append(r)
    agg = defaultdict(list)       # (transport, L) -> [(offered/GPU, node achieved, worst p99, min success, cores)]
    for (t, L, q), rs in pts.items():
        if len(rs) < max(r["gpus"] for r in rs):
            continue              # incomplete point
        agg[(t, L)].append((q, sum(r["achieved"] for r in rs), max(r["p99_ms"] for r in rs),
                            min(r["success"] for r in rs), sum(r["cores_other"] for r in rs), len(rs)))
    Ls = sorted({k[1] for k in agg})
    trs = [t for t in ORDER if any(k[0] == t for k in agg)]
    cap = 3 * a.slo
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.4 * len(Ls), 3.4), dpi=140, facecolor=SURFACE, squeeze=False)
    for ax, L in zip(axes[0], Ls):
        ax.set_facecolor(SURFACE)
        for t in trs:
            p = sorted(agg.get((t, L), []))
            if not p:
                continue
            x = [8 * v[0] for v in p]
            ax.plot(x, [min(v[2], cap) for v in p], color=COLORS[t], marker=MARK[t], ms=4.5, lw=1.8, mec=SURFACE,
                    mew=1, label=NAMES[t], zorder=3)
        ax.axhline(a.slo, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(0.02, a.slo / cap + 0.015, f"SLO {a.slo:.0f} ms", transform=ax.transAxes, color=INK2, fontsize=7.5)
        ax.set_ylim(0, cap * 1.04)
        ax.set_xlim(left=0)
        ax.set_title(f"history {L // 1024}k tokens", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("offered load, whole node (users/s, 8 GPUs)", color=INK2, fontsize=8.5)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("worst GPU's p99 latency (ms)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=len(trs), frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.01, 1.08))
    fig.suptitle("Config M, 256 candidates, all 8 GPUs of the inference node serving at once (each GPU: own NIC, own "
                 "store rank)", color=INK, fontsize=10, y=1.15, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.md:
        print(f"\n**Node throughput (users/s, 8 GPUs) at the highest offered rate where every GPU has p99 ≤ "
              f"{a.slo:.0f} ms and ≥ 99% of requests within it; transport CPU cores on the node there**\n")
        print("| Transport | " + " | ".join(f"{L // 1024}k: users/s | cores" for L in Ls) + " |")
        print("|---|" + "---|---|" * len(Ls))
        for t in trs:
            cells = []
            for L in Ls:
                ok = [v for v in agg.get((t, L), []) if v[2] <= a.slo and v[3] >= 0.99]
                if ok:
                    v = max(ok)
                    cells += [f"**{v[1]:,.0f}**", f"{v[4]:.1f}"]
                else:
                    cells += ["–", "–"]
            print(f"| {NAMES[t]} | " + " | ".join(cells) + " |")
        print("\n**Saturation: highest node throughput reached at any offered rate (users/s)**\n")
        print("| Transport | " + " | ".join(f"{L // 1024}k" for L in Ls) + " |")
        print("|---|" + "---|" * len(Ls))
        for t in trs:
            print(f"| {NAMES[t]} | " + " | ".join(f"{max((v[1] for v in agg.get((t, L), [])), default=0):,.0f}" for L in Ls) + " |")


if __name__ == "__main__":
    main()
