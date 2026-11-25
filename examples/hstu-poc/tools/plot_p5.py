#!/usr/bin/env python3
"""P5 serving curves: per history length, p99 latency and the share of requests within the SLO vs the OFFERED load
(each x has one point; past saturation the achieved rate stops growing, so plotting against it folds the curve).
    tools/plot_p5.py results/p5b/*.jsonl -o docs/figs/p5-curves.png [--slo 100] [--fetch raw] [--md]
Series: remote fetch (one transport, --fetch; the four coincide on one GPU) and recompute. p99 is clipped at 3 x SLO."""
import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOTS = {"fetch": "#2a78d6", "recompute": "#eb6834"}
MARK = {"fetch": "o", "recompute": "s"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAMES = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "verbs": "CPU verbs", "proxy": "GIN-Proxy"}


def load(paths, gpus=1):
    rows = [json.loads(x) for p in paths for f in glob.glob(p) for x in open(f) if x.strip()]
    return [r for r in rows if r.get("gpus", 1) == gpus and not r.get("hog_ms")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--slo", type=float, default=100.0)
    ap.add_argument("--fetch", default="raw", help="transport drawn as 'remote fetch'")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = load(a.paths)
    by = defaultdict(list)
    for r in rows:
        by[(r["L"], r["strategy"])].append(r)
    Ls = sorted({r["L"] for r in rows})
    cap = 3 * a.slo
    fig, axes = plt.subplots(2, len(Ls), figsize=(3.3 * len(Ls), 5.2), dpi=140, facecolor=SURFACE, squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, L in enumerate(Ls):
        ax, ax2 = axes[0][j], axes[1][j]
        for s, key in (("fetch", a.fetch), ("recompute", "recompute")):
            pts = sorted(by.get((L, key), []), key=lambda r: r["offered"])
            if not pts:
                continue
            x = [r["offered"] for r in pts]
            y = [min(r["p99_ms"], cap) for r in pts]
            lab = f"remote fetch ({NAMES.get(key, key)})" if s == "fetch" else "recompute"
            ax.plot(x, y, color=SLOTS[s], lw=1.8, marker=MARK[s], ms=4.5, mec=SURFACE, mew=1, label=lab, zorder=3)
            clipped = [(xi, cap) for xi, r in zip(x, pts) if r["p99_ms"] > cap]
            if clipped:
                ax.scatter(*zip(*clipped), marker="^", s=40, color=SLOTS[s], edgecolor=SURFACE, zorder=4)
            ax2.plot(x, [100 * r["success"] for r in pts], color=SLOTS[s], lw=1.8, marker=MARK[s], ms=4.5,
                     mec=SURFACE, mew=1, zorder=3)
        ax.axhline(a.slo, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(0.02, a.slo / cap + 0.015, f"SLO {a.slo:.0f} ms", transform=ax.transAxes, color=INK2, fontsize=7.5)
        ax.set_ylim(0, cap * 1.04)
        ax.set_title(f"history {L // 1024}k tokens", color=INK, fontsize=10, loc="left")
        ax2.set_ylim(0, 103)
        ax2.axhline(99, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax2.set_xlabel("offered load (users/s, 1 GPU)", color=INK2, fontsize=8.5)
        for z in (ax, ax2):
            z.set_facecolor(SURFACE)
            z.grid(True, color=GRID, lw=0.6, zorder=0)
            z.tick_params(colors=INK2, labelsize=7.5)
            z.set_xlim(left=0)
            for sp in z.spines.values():
                sp.set_color(GRID)
    axes[0][0].set_ylabel(f"p99 latency (ms; ▲ = above {cap:.0f})", color=INK2, fontsize=8.5)
    axes[1][0].set_ylabel(f"requests within {a.slo:.0f} ms (%)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.01, 1.05))
    fig.suptitle("Config M, 256 candidates, 1 GPU, open-loop Poisson load: p99 latency and SLO success vs offered load",
                 color=INK, fontsize=10, y=1.09, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.md:
        strategies = sorted({r["strategy"] for r in rows}, key=lambda s: (s == "recompute", s))
        print(f"\n**Max offered load (users/s, 1 GPU) with p99 ≤ {a.slo:.0f} ms and ≥ 99% of requests within it; "
              "in brackets: p50 / p99 ms there; transport CPU cores**\n")
        print("| Strategy | " + " | ".join(f"{L // 1024}k" for L in Ls) + " |")
        print("|---|" + "---|" * len(Ls))
        for s in strategies:
            cells = []
            for L in Ls:
                ok = [r for r in by.get((L, s), []) if r["p99_ms"] <= a.slo and r["success"] >= 0.99]
                if ok:
                    r = max(ok, key=lambda r: r["offered"])
                    cells.append(f"**{r['offered']:,.0f}** ({r['p50_ms']:.1f} / {r['p99_ms']:.0f}; "
                                 f"{r['cores_other']:.2f} cores)")
                else:
                    cells.append("–")
            print(f"| {NAMES.get(s, s)} | " + " | ".join(cells) + " |")
        n_req = sum(r["n"] for r in rows if r["strategy"] != "recompute")
        n_chk = sum(r.get("checks", 0) for r in rows)
        n_bad = sum(r.get("bad_words", 0) for r in rows)
        print(f"\nData checks: {n_chk} sampled batches, {n_bad} mismatched words; {n_req:,} fetched requests served.")


if __name__ == "__main__":
    main()
