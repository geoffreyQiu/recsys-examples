#!/usr/bin/env python3
"""P2t report: the fastest RDMA implementation per request size (plain-run latency, host clock), from results/p2t.
    tools/p2t_report.py results/p2t --best docs/figs/best-impl.png --ablation docs/figs/opt-ablation.png \
        --bound docs/figs/rdma-vs-bound.png [--md]
Modes: transport.level.variant (transport verbs|raw|gdaki|proxy; level b = basic implementation, t = every latency
option, zc / spin / ac = one option alone, b2 = basic again for run-to-run noise; variant 5a|pdl|pdl_early|fused),
plus recompute and local. Latency = host-clock mean (request issued -> output seen complete)."""
import argparse
import glob
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TNAME = {"verbs": "CPU verbs", "raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "proxy": "GIN-Proxy"}
TSHORT = {"verbs": "verbs", "raw": "raw", "gdaki": "GIN-G", "proxy": "GIN-P"}
VNAME = {"5a": "wait kernel", "pdl": "PDL", "pdl_early": "PDL early release", "fused": "fused wait", "tile": "per-tile wait"}
VSHORT = {"5a": "wait", "pdl": "pdl", "pdl_early": "early", "fused": "fused", "tile": "tile"}
TCOL = {"verbs": "#2a78d6", "raw": "#1baf7a", "gdaki": "#6250d6", "proxy": "#eda100"}
KV = {"S": 6 * 1024, "M": 32 * 1024, "L": 64 * 1024}
LAYERS = {"S": 3, "M": 8, "L": 8}
BASE = "verbs.b.5a"


def load(d):
    rows = []
    for f in glob.glob(f"{d}/*.jsonl"):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    lat, bad = {}, 0
    for r in rows:
        if "skipped" in r or "host_mean_ms" not in r:
            continue
        lat[(r["mode"], r["config"], int(r["B"]), int(r["L"]))] = r["host_mean_ms"]
        bad += int(r.get("check_bad_words") or 0)
    return lat, bad


def rdma_modes(lat):
    return sorted({k[0] for k in lat if k[0].count(".") == 2 and not k[0].split(".")[1] == "b2"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--best", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--bound", required=True)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    lat, bad = load(a.dir)
    modes = rdma_modes(lat)
    cells = sorted({k[1:] for k in lat if k[0] == BASE})
    # run-to-run noise: the basic CPU-verbs run repeated (b2) on the small grid
    rep = [abs(lat[("verbs.b2.5a",) + c] / lat[(BASE,) + c] - 1) for c in cells if ("verbs.b2.5a",) + c in lat]
    noise = float(np.percentile(rep, 95)) if rep else 0.01
    # best RDMA per cell, over every mode measured at that cell
    best = {}
    for c in cells:
        cand = [(lat[(m,) + c], m) for m in modes if (m,) + c in lat]
        if cand:
            best[c] = min(cand)
    cfgs = [c for c in ("S", "M", "L") if any(x[0] == c for x in cells)]

    # ---- figure 1: best implementation map (small grid: every mode measured) ----
    small = [c for c in cells if sum(1 for m in modes if (m,) + c in lat) >= 10]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.6 * len(cfgs), 3.4), dpi=140, facecolor=SURFACE, squeeze=False)
    for ax, cf in zip(axes[0], cfgs):
        ax.set_facecolor(SURFACE)
        Ls = sorted({c[2] for c in small if c[0] == cf})
        Bs = sorted({c[1] for c in small if c[0] == cf})
        M = np.full((len(Bs), len(Ls)), np.nan)
        for i, B in enumerate(Bs):
            for j, L in enumerate(Ls):
                c = (cf, B, L)
                if c in best and (BASE,) + c in lat:
                    M[i, j] = 100 * (1 - best[c][0] / lat[(BASE,) + c])
        ax.imshow(M, cmap="Greens", vmin=0, vmax=8, aspect="auto", origin="lower")
        for i, B in enumerate(Bs):
            for j, L in enumerate(Ls):
                c = (cf, B, L)
                if c not in best or np.isnan(M[i, j]):
                    continue
                t, lv, v = best[c][1].split(".")
                txt = f"{TSHORT[t]}\n{VSHORT[v]}" if M[i, j] / 100 > noise else "≈ basic"
                ax.text(j, i, f"{txt}\n{M[i, j]:+.0f}%" if M[i, j] / 100 > noise else txt, ha="center", va="center",
                        fontsize=6.2, color=INK if M[i, j] < 5 else SURFACE)
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels([f"{L / 1024:g}k" for L in Ls], fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(Bs)))
        ax.set_yticklabels(Bs, fontsize=7.5, color=INK2)
        ax.set_xlabel("history length (tokens)", color=INK2, fontsize=8.5)
        ax.set_title(f"config {cf}", color=INK, fontsize=10, loc="left")
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("batch size B (users)", color=INK2, fontsize=8.5)
    fig.suptitle(f"Fastest RDMA implementation per request (transport + wait variant, all latency options) and its gain "
                 f"over basic CPU verbs; '≈ basic' = within run-to-run noise ({100 * noise:.1f}%)",
                 color=INK, fontsize=9.5, y=1.04, x=0.01, ha="left")
    fig.savefig(a.best, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.best)

    # ---- figure 2: ablation on the smallest requests (0.5k history, one user), where the options matter ----
    reps = [c for c in (("S", 1, 512), ("M", 1, 512), ("L", 1, 512)) if (BASE,) + c in lat]
    trs = [t for t in ("verbs", "raw", "gdaki", "proxy") if any(m.startswith(t + ".") for m in modes)]
    order = ["b.5a", "zc.5a", "spin.5a", "ac.5a", "t.5a", "t.pdl", "t.pdl_early", "t.fused", "t.tile"]
    oname = {"b.5a": "basic", "zc.5a": "zero-copy descriptor only", "spin.5a": "spinning worker only",
             "ac.5a": "adaptive contexts only", "t.5a": "all options, wait kernel", "t.pdl": "all options, PDL",
             "t.pdl_early": "all options, PDL early release", "t.fused": "all options, fused wait",
             "t.tile": "all options, per-tile wait"}
    rowsx = [(t, o) for t in trs for o in order if any((f"{t}.{o}",) + c in lat for c in reps)]
    fig, axes = plt.subplots(1, len(reps), figsize=(3.3 * len(reps) + 1.8, 0.24 * len(rowsx) + 1.2), dpi=140,
                             facecolor=SURFACE, squeeze=False, sharey=True)
    ys = np.arange(len(rowsx))[::-1]
    for ax, c in zip(axes[0], reps):
        ax.set_facecolor(SURFACE)
        base = lat[(BASE,) + c]
        for y, (t, o) in zip(ys, rowsx):
            m = f"{t}.{o}"
            if (m,) + c not in lat:
                continue
            v = 100 * (lat[(m,) + c] / base - 1)
            ax.barh(y, v, height=0.72, color=TCOL[t], alpha=0.45 if o == "b.5a" else 1.0, zorder=3)
            ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=6.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvspan(-100 * noise, 100 * noise, color=GRID, alpha=0.7, zorder=1)
        ax.set_title(f"config {c[0]}  (basic verbs {base * 1e3:.0f} µs)", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("latency vs basic CPU verbs (%)", color=INK2, fontsize=8)
        ax.grid(True, axis="x", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -9.5), hi + 3)
    axes[0][0].set_yticks(ys)
    axes[0][0].set_yticklabels([f"{TNAME[t]}: {oname[o]}" for t, o in rowsx], fontsize=7.5, color=INK)
    fig.suptitle("Latency options and wait variants on the smallest requests (0.5k history, one user); grey band = "
                 "run-to-run noise", color=INK, fontsize=9.5, y=1.0, x=0.01, ha="left")
    fig.savefig(a.ablation, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.ablation)

    # ---- figure 3: distance to the lower bound, basic vs best ----
    big = [c[1] * c[2] * KV[c[0]] / (best[c][0] / 1e3) for c in cells if c in best and c[1] * c[2] * KV[c[0]] >= 4e9]
    bw = float(np.median(big)) if big else 49.4e9          # measured line rate (largest requests)
    print(f"measured line rate (median over requests >= 4 GB): {bw / 1e9:.2f} GB/s")
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=140, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for lab, fn, col, mk in (("basic CPU verbs", lambda c: lat.get((BASE,) + c), MUTED, "o"),
                             ("best RDMA implementation", lambda c: best.get(c, (None,))[0], "#1baf7a", "^")):
        xs, ys = [], []
        for c in cells:
            v, loc = fn(c), lat.get(("local",) + c)
            if v is None or loc is None:
                continue
            cf, B, L = c
            lb = max(loc, B * L * KV[cf] / bw * 1e3)
            xs.append(B * L * KV[cf])
            ys.append(100 * (v / lb - 1))
        ax.scatter(xs, ys, s=9, color=col, marker=mk, alpha=0.8, lw=0, label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("bytes fetched per request", color=INK2, fontsize=8.5)
    ax.set_ylabel("latency above the lower bound (%)", color=INK2, fontsize=8.5)
    ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title(f"Remote fetch above its lower bound max(local-hit latency, bytes ÷ {bw / 1e9:.1f} GB/s)", color=INK,
                 fontsize=9.5, loc="left")
    fig.savefig(a.bound, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.bound)

    if not a.md:
        return
    print(f"\nData checks: {bad} mismatched words. Run-to-run noise (basic CPU verbs repeated): median "
          f"{100 * np.median(rep):.2f}%, 95th percentile {100 * noise:.2f}%, max {100 * max(rep):.2f}% over {len(rep)} cells")
    # gains of the best implementation over basic verbs, and of each transport's best over its basic
    print("\n**Gain of the fastest implementation over basic CPU verbs (small grid; % of latency; '·' = within noise)**\n")
    for cf in cfgs:
        Ls = sorted({c[2] for c in small if c[0] == cf})
        Bs = sorted({c[1] for c in small if c[0] == cf})
        print(f"\n{cf}: | B \\ L | " + " | ".join(f"{L / 1024:g}k" for L in Ls) + " |")
        print("|---|" + "---|" * len(Ls))
        for B in Bs:
            cellstr = []
            for L in Ls:
                c = (cf, B, L)
                g = 1 - best[c][0] / lat[(BASE,) + c]
                t, lv, v = best[c][1].split(".")
                cellstr.append(f"{100 * g:+.1f}% {TSHORT[t]}/{lv}/{VSHORT[v]}" if g > noise else "·")
            print(f"| {B} | " + " | ".join(cellstr) + " |")
    print("\n**Per transport: tuned (best wait variant) vs its own basic implementation, small grid**\n")
    print("| Transport | median gain | max gain (where) | best variant most often |")
    print("|---|---|---|---|")
    for t in trs:
        gains, wins = [], defaultdict(int)
        for c in small:
            if (f"{t}.b.5a",) + c not in lat:
                continue
            tv = [(lat[(f"{t}.t.{v}",) + c], v) for v in VSHORT if (f"{t}.t.{v}",) + c in lat]
            if not tv:
                continue
            bt, bv = min(tv)
            gains.append((1 - bt / lat[(f"{t}.b.5a",) + c], c))
            wins[bv] += 1
        if gains:
            g = sorted(gains)
            print(f"| {TNAME[t]} | {100 * np.median([x[0] for x in g]):+.1f}% | {100 * g[-1][0]:+.1f}% "
                  f"({g[-1][1][0]}, B={g[-1][1][1]}, L={g[-1][1][2]}) | {max(wins, key=wins.get)} "
                  f"({dict(wins)}) |")
    print("\n**Wait variants (all options on): change vs the wait kernel, per transport (small grid)**\n")
    print("| Transport | PDL | PDL early release | fused wait | per-tile wait |")
    print("|---|---|---|---|---|")
    for t in trs:
        cols = []
        for v in ("pdl", "pdl_early", "fused", "tile"):
            ch = [lat[(f"{t}.t.{v}",) + c] / lat[(f"{t}.t.5a",) + c] - 1 for c in small
                  if (f"{t}.t.{v}",) + c in lat and (f"{t}.t.5a",) + c in lat]
            cols.append(f"{100 * min(ch):+.1f} … {100 * max(ch):+.1f}% (median {100 * np.median(ch):+.1f}%)" if ch else "–")
        print(f"| {TNAME[t]} | " + " | ".join(cols) + " |")
    # overall ranking of combinations: how often each is fastest or within noise of the fastest
    print("\n**How often each combination is the fastest or within noise of it (small grid cells)**\n")
    score = defaultdict(int)
    for c in small:
        bt = best[c][0]
        for m in modes:
            if (m,) + c in lat and lat[(m,) + c] <= bt * (1 + noise):
                score[m] += 1
    for m, n in sorted(score.items(), key=lambda x: -x[1])[:12]:
        print(f"- {m}: {n} of {len(small)}")
    # break-even with the best RDMA vs recompute, per config and B (every cell measured with both)
    print("\n**Latency break-even vs recompute: basic CPU verbs / best RDMA**\n")
    for cf in cfgs:
        for B in sorted({c[1] for c in cells if c[0] == cf}):
            Ls = sorted({c[2] for c in cells if c[0] == cf and c[1] == B})
            out = []
            for fn in (lambda c: lat.get((BASE,) + c), lambda c: best.get(c, (None,))[0]):
                pts = [(L, lat[("recompute", cf, B, L)] / fn((cf, B, L))) for L in Ls
                       if ("recompute", cf, B, L) in lat and fn((cf, B, L))]
                be = None
                for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
                    if r0 < 1 <= r1:
                        be = math.exp(math.log(l0) + (0 - math.log(r0)) * (math.log(l1) - math.log(l0)) /
                                      (math.log(r1) - math.log(r0)))
                out.append(f"{be / 1024:.1f}k" if be else "–")
            print(f"- {cf} B={B}: {out[0]} / {out[1]}")


if __name__ == "__main__":
    main()
