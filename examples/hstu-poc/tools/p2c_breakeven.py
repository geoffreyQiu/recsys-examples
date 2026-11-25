#!/usr/bin/env python3
"""P2c: fetch vs recompute break-even for workload R (history fully cached remotely), from results/p2c/.
    tools/p2c_breakeven.py results/p2c -o docs/figs/p2c-breakeven.png [--md]
Latency break-even, per (config, B): the history length L* where the best transport's mean request latency equals
recompute's (linear interpolation of log(ratio) over log(L) between the two grid points that bracket the crossing).
Throughput break-even, per config: the L where a pipelined server's ceiling is equal for both strategies
(fetch: min(NIC rate / KV bytes per user, best-B local-hit compute rate); recompute: best-B compute rate)."""
import argparse
import glob
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

TRANSPORTS = ["raw", "gdaki", "verbs", "proxy"]
NAMES = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "verbs": "CPU verbs", "proxy": "GIN-Proxy"}
KV_TOK = {"S": 3 * 2 * 512 * 2, "M": 8 * 2 * 1024 * 2, "D": 16 * 2 * 1024 * 2, "L": 8 * 2 * 2048 * 2}   # KV bytes/token, all layers
NIC_GBS = 49.4
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RED = "#2a78d6", "#e34948"
# diverging: blue arm (fetch faster) <- gray midpoint -> red arm (recompute faster)
DIV = LinearSegmentedColormap.from_list("div", ["#9e2b2a", "#e34948", "#f3aaa9", "#f0efec", "#9ec5f4", "#3987e5", "#104281"])


def load(d):
    rows = []
    for f in glob.glob(f"{d}/*.jsonl"):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    lat = defaultdict(dict)        # (config, B, L) -> {mode: mean_ms}
    load.rows = [r for r in rows if "skipped" not in r and str(r["L"]) != "skew"]
    for r in rows:
        if "skipped" in r or str(r["L"]) == "skew":
            continue
        lat[(r["config"], int(r["B"]), int(r["L"]))][r["mode"]] = r["mean_ms"]
    return lat


def best_fetch(m):
    t = [(m[k], k) for k in TRANSPORTS if k in m]
    return min(t) if t else (None, None)


def crossing(Ls, ratio):
    """L* above which ratio = recompute / fetch stays >= 1 (fetch faster): the LAST upward crossing, by log-log
    interpolation (a near-tie at a short length, e.g. 1.003 at 1k, is not the break-even)."""
    pts = [(l, r) for l, r in zip(Ls, ratio) if r is not None]
    if not pts or pts[-1][1] < 1:
        return None                  # recompute still faster at the longest length
    for (l0, r0), (l1, r1) in reversed(list(zip(pts, pts[1:]))):
        if r0 < 1 <= r1:
            x0, x1, y0, y1 = math.log(l0), math.log(l1), math.log(r0), math.log(r1)
            return math.exp(x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
    return float(pts[0][0])          # fetch faster at every length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--md", action="store_true", help="print markdown tables")
    ap.add_argument("--configs", default="S,M,L")
    a = ap.parse_args()
    lat = load(a.dir)
    configs = [c for c in a.configs.split(",") if any(k[0] == c for k in lat)]
    Bs = sorted({k[1] for k in lat})
    Ls = sorted({k[2] for k in lat})
    fig, axes = plt.subplots(1, len(configs), figsize=(4.6 * len(configs), 3.5), dpi=140, facecolor=SURFACE,
                             squeeze=False)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)       # log2(recompute / fetch): +-2x
    summary = {}
    for ax, c in zip(axes[0], configs):
        M = np.full((len(Bs), len(Ls)), np.nan)
        cross = {}
        for i, B in enumerate(Bs):
            ratios = []
            for j, L in enumerate(Ls):
                m = lat.get((c, B, L), {})
                f, _ = best_fetch(m)
                r = m.get("recompute")
                ratios.append(r / f if (f and r) else None)
                if f and r:
                    M[i, j] = math.log2(r / f)
            cross[B] = crossing(Ls, ratios)
        # throughput ceilings (users/s): fetch = min(NIC, local compute), recompute = best B
        tput = {}
        for L in Ls:
            nic = NIC_GBS * 1e9 / (L * KV_TOK[c])
            loc = max([B / (lat[(c, B, L)]["local"] / 1e3) for B in Bs if "local" in lat.get((c, B, L), {})] or [0])
            rec = max([B / (lat[(c, B, L)]["recompute"] / 1e3) for B in Bs if "recompute" in lat.get((c, B, L), {})] or [0])
            tput[L] = (min(nic, loc) if loc else nic, rec)
        t_ratio = [tput[L][1] / tput[L][0] if tput[L][1] else None for L in Ls]
        # recompute faster -> ratio > 1 here, so cross where fetch/recompute ceiling ratio goes above 1
        t_cross = crossing(Ls, [1 / x if x else None for x in t_ratio])
        summary[c] = (cross, t_cross, tput)
        ax.set_facecolor(SURFACE)
        x = np.arange(len(Ls) + 1) - 0.5
        y = np.arange(len(Bs) + 1) - 0.5
        ax.pcolormesh(x, y, M, cmap=DIV, norm=norm, edgecolors=SURFACE, linewidth=1.5)
        # latency break-even line: per B, fractional column of L*
        lx = lambda L: np.interp(math.log(L), [math.log(v) for v in Ls], np.arange(len(Ls)))  # noqa: E731
        pts = [(lx(cross[B]), i) for i, B in enumerate(Bs) if cross[B]]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=INK, lw=2, marker="o", ms=5, mec=SURFACE,
                    mew=1.2, zorder=4, label="latency break-even (per batch size)")
        if t_cross:
            ax.axvline(lx(t_cross), color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3, label="throughput break-even")
        vals = [v for v in cross.values() if v]
        if vals:
            ax.text(0.99, 1.02, f"L* = {min(vals) / 1024:.1f}k–{max(vals) / 1024:.1f}k", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels([f"{L / 1024:g}k" for L in Ls], fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(Bs)))
        ax.set_yticklabels([str(B) for B in Bs], fontsize=8, color=INK2)
        ax.set_xlabel("history length L (tokens)", color=INK2, fontsize=8.5)
        ax.set_title(f"config {c}", color=INK, fontsize=10, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("batch size B (users)", color=INK2, fontsize=8.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIV)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(["recompute 2×\nfaster", "1.4×", "equal", "1.4×", "fetch 2×\nfaster"], fontsize=7.5, color=INK2)
    cb.outline.set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.01, 1.07))
    fig.suptitle("Recompute latency ÷ best remote-fetch latency (isolated requests, 1 GPU, 256 candidates; blank = does not fit in memory)",
                 color=INK, fontsize=10, y=1.13, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.md:
        print("\n| Config | " + " | ".join(f"B={B}" for B in Bs) + " | throughput break-even |")
        print("|---|" + "---|" * (len(Bs) + 1))
        for c in configs:
            cross, t_cross, _ = summary[c]
            fmt = lambda v: f"{v / 1024:.1f}k" if v else "> 32k"  # noqa: E731
            print(f"| {c} | " + " | ".join(fmt(cross[B]) for B in Bs) + f" | {fmt(t_cross)} |")
        for c in configs:
            print(f"\n**{c}: mean latency (ms), recompute / best fetch (transport)**\n")
            print("| L \\ B | " + " | ".join(str(B) for B in Bs) + " |")
            print("|---|" + "---|" * len(Bs))
            for L in Ls:
                cells = []
                for B in Bs:
                    m = lat.get((c, B, L), {})
                    f, k = best_fetch(m)
                    r = m.get("recompute")
                    cells.append(f"{r:.3g} / {f:.3g}" if (f and r) else (f"– / {f:.3g}" if f else (f"{r:.3g} / –" if r else "–")))
                print(f"| {L / 1024:g}k | " + " | ".join(cells) + " |")
            print(f"\n**{c}: spread of the four transports (max/min − 1 of the means, %)**\n")
            print("| L \\ B | " + " | ".join(str(B) for B in Bs) + " |")
            print("|---|" + "---|" * len(Bs))
            for L in Ls:
                cells = []
                for B in Bs:
                    m = lat.get((c, B, L), {})
                    v = [m[k] for k in TRANSPORTS if k in m]
                    cells.append(f"{100 * (max(v) / min(v) - 1):.1f}" if len(v) > 1 else "–")
                print(f"| {L / 1024:g}k | " + " | ".join(cells) + " |")
        print("\n**Sensitivity: latency break-even L* if the NIC were faster (estimate: the transfer part of each fetch "
              "measurement, bytes ÷ 49.4 GB/s, scaled down; the rest of the request kept)**\n")
        print("| Config | NIC | " + " | ".join(f"B={B}" for B in Bs) + " |")
        print("|---|---|" + "---|" * len(Bs))
        for c in configs:
            for sc in (1.0, 1.5, 2.0):
                cells = []
                for B in Bs:
                    ratios = []
                    for L in Ls:
                        m = lat.get((c, B, L), {})
                        f, _ = best_fetch(m)
                        r = m.get("recompute")
                        if not (f and r):
                            ratios.append(None)
                            continue
                        tr = B * L * KV_TOK[c] / (NIC_GBS * 1e9) * 1e3
                        loc = m.get("local", 0.0)
                        fs = max(f - tr + tr / sc, loc)
                        ratios.append(r / fs)
                    v = crossing(Ls, ratios)
                    cells.append(f"{v / 1024:.1f}k" if v else "> 32k")
                print(f"| {c} | {sc:g}× ({NIC_GBS * sc * 8:.0f} Gb/s) | " + " | ".join(cells) + " |")
        rows = load.rows
        print("\n**Recompute efficiency η (achieved ÷ 989 TFLOPS BF16)**\n")
        print("| Config, L \\ B | " + " | ".join(str(B) for B in Bs) + " |")
        print("|---|" + "---|" * len(Bs))
        eta = {(r["config"], int(r["B"]), int(r["L"])): r["eta"] for r in rows if r["mode"] == "recompute"}
        for c in configs:
            for L in Ls:
                print(f"| {c}, {L / 1024:g}k | " + " | ".join(f"{100 * eta[(c, B, L)]:.0f}%" if (c, B, L) in eta else "–" for B in Bs) + " |")
        dev = sorted(abs(r["mean_ms"] / r["p50_ms"] - 1) for r in rows)
        big = [r for r in rows if abs(r["mean_ms"] / r["p50_ms"] - 1) > 0.01]
        print(f"\nmean vs p50 over {len(dev)} cells: median |mean/p50 - 1| {100 * dev[len(dev) // 2]:.2f}%, "
              f"95th percentile {100 * dev[int(0.95 * len(dev))]:.2f}%, max {100 * dev[-1]:.1f}%; cells > 1%: {len(big)} "
              + ", ".join(f"{r['config']} {r['mode']} B={r['B']} L={r['L']} ({100 * (r['mean_ms'] / r['p50_ms'] - 1):+.1f}%)" for r in big[:8]))
        print("\n**Throughput ceilings per GPU (users/s): fetch min(NIC, local compute) / recompute (best B)**\n")
        print("| L | " + " | ".join(configs) + " |")
        print("|---|" + "---|" * len(configs))
        for L in Ls:
            print(f"| {L / 1024:g}k | " + " | ".join(f"{summary[c][2][L][0]:,.0f} / {summary[c][2][L][1]:,.0f}" for c in configs) + " |")


if __name__ == "__main__":
    main()
