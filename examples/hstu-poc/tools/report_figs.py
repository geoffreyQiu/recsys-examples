#!/usr/bin/env python3
"""Report figures from the isolated-request grid (results/p2c):
  cost  -> time per user vs history length: recompute and remote fetch, single-user requests and batches of 16
           (local hit in grey for reference)
  xport -> each RDMA transport's request latency relative to CPU verbs (the RDMA baseline) vs bytes moved
    tools/report_figs.py results/p2c --cost docs/figs/cost-per-user.png --xport docs/figs/transport-overhead.png"""
import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
RECOMP, FETCH = "#eb6834", "#2a78d6"
XCOL = {"raw": ("#1baf7a", "^", "raw GDAKI"), "gdaki": ("#6250d6", "D", "GIN-GDAKI"), "proxy": ("#eda100", "v", "GIN-Proxy")}
T = ["verbs", "raw", "gdaki", "proxy"]
KV_TOK = {"S": 3 * 2 * 512 * 2, "M": 8 * 2 * 1024 * 2, "L": 8 * 2 * 2048 * 2, "D": 16 * 2 * 1024 * 2}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--xport", required=True)
    ap.add_argument("--configs", default="M,L")
    ap.add_argument("--bbig", type=int, default=16)
    a = ap.parse_args()
    lat = {}
    for f in glob.glob(f"{a.dir}/*.jsonl"):
        for x in open(f):
            r = json.loads(x)
            if "skipped" not in r and str(r["L"]) != "skew":
                lat[(r["mode"], r["config"], int(r["B"]), int(r["L"]))] = r["mean_ms"]
    cfgs = a.configs.split(",")
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.2 * len(cfgs), 3.8), dpi=140, facecolor=SURFACE, squeeze=False)
    for ax, c in zip(axes[0], cfgs):
        style(ax)
        Ls = sorted({k[3] for k in lat if k[1] == c})
        for B, ls, lw in ((1, "-", 1.8), (a.bbig, (0, (5, 2)), 1.8)):
            def series(mode_fn):
                xs, ys = [], []
                for L in Ls:
                    v = mode_fn(B, L)
                    if v is not None:
                        xs.append(L)
                        ys.append(v / B)
                return xs, ys
            best = lambda B_, L: min([lat[(t, c, B_, L)] for t in T if (t, c, B_, L) in lat] or [None]) \
                if any((t, c, B_, L) in lat for t in T) else None  # noqa: E731
            xs, ys = series(lambda B_, L: lat.get(("recompute", c, B_, L)))
            ax.plot(xs, ys, color=RECOMP, ls=ls, lw=lw, marker="s", ms=4, mec=SURFACE, mew=0.8, zorder=3,
                    label=f"recompute, B = {B}" + (" (per user)" if B > 1 else ""))
            xs, ys = series(best)
            ax.plot(xs, ys, color=FETCH, ls=ls, lw=lw, marker="o", ms=4, mec=SURFACE, mew=0.8, zorder=3,
                    label=f"remote fetch, B = {B}" + (" (per user)" if B > 1 else ""))
            xs, ys = series(lambda B_, L: lat.get(("local", c, B_, L)))
            if B > 1:
                ax.plot(xs, ys, color=MUTED, ls=ls, lw=1.2, zorder=2, label=f"local hit, B = {B} (reference)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        tk = [L for L in Ls if L in (512, 1024, 2048, 4096, 8192, 16384, 32768)]
        ax.set_xticks(tk)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([f"{L / 1024:g}k" for L in tk], fontsize=7.5, color=INK2)
        ax.set_xlabel("history length L (tokens)", color=INK2, fontsize=8.5)
        ax.set_title(f"config {c}", color=INK, fontsize=10, loc="left")
    axes[0][0].set_ylabel("request latency ÷ B (ms per user)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.12))
    fig.suptitle("Cost of one user's request: recompute the history vs fetch its K/V (fastest transport), 256 candidates",
                 color=INK, fontsize=10, y=1.17, x=0.01, ha="left")
    fig.savefig(a.cost, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.cost)

    # transport overhead vs bytes moved, relative to CPU verbs (median per bytes bin, all configs and batch sizes)
    pts = defaultdict(list)
    for (m, c, B, L), v in lat.items():
        if m in XCOL and c in ("S", "M", "L") and ("verbs", c, B, L) in lat:
            pts[m].append((B * L * KV_TOK[c], 100 * (v / lat[("verbs", c, B, L)] - 1)))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=140, facecolor=SURFACE)
    style(ax)
    bins = np.logspace(np.log10(1e6), np.log10(4e10), 22)
    for m, (col, mk, name) in XCOL.items():
        p = np.array(pts.get(m, []))
        if not len(p):
            continue
        idx = np.digitize(p[:, 0], bins)
        xs, ys = [], []
        for k in sorted(set(idx)):
            sel = p[idx == k]
            xs.append(np.median(sel[:, 0]))
            ys.append(np.median(sel[:, 1]))
        ax.scatter(p[:, 0], p[:, 1], s=7, color=col, alpha=0.25, lw=0, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.8, marker=mk, ms=4, mec=SURFACE, mew=0.8, zorder=3, label=name)
    ax.axhline(0, color=INK2, lw=1)
    ax.text(1.2e6, 1.2, "CPU verbs (baseline)", color=INK2, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("bytes fetched per request (B × L × KV bytes per token)", color=INK2, fontsize=8.5)
    ax.set_ylabel("latency vs CPU verbs (%)", color=INK2, fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("RDMA mechanisms: per-request overhead vs CPU verbs (isolated requests; S, M, L; B = 1-32)",
                 color=INK, fontsize=9.5, loc="left")
    fig.savefig(a.xport, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.xport)


if __name__ == "__main__":
    main()
