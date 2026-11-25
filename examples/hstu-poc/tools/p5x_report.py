#!/usr/bin/env python3
"""Serving capacity at relaxed SLOs and up to saturation (results/p5s strict-SLO rows + results/p5x saturation rows).
    tools/p5x_report.py results/p5s results/p5x -o docs/figs/slo-breakeven.png [--md]
A load point meets a p99 target X when p99 <= X (i.e. >= 99% of requests within X), it has no data error and >= 90%
of the offered load was achieved. 'sustained' = the highest load the server keeps up with: achieved >= 97% of offered
and p99 <= 2 s (beyond it the queue grows and requests time out). Capacity = the highest offered load that passes.
Each level uses one campaign, so both strategies were swept alike: 100 ms from P5s (recompute: dynamic batching with
a 50 ms cap or one request at a time, the better one; fetch: CPU verbs, basic), the relaxed levels from P5x
(recompute: dynamic batching with a 500 ms cap or one request at a time; fetch: raw GDAKI with the fused wait; basic
CPU verbs at 1k / 4k / 16k for comparison)."""
import argparse
import glob
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LEVELS = [("100", 100.0), ("200", 200.0), ("500", 500.0), ("1000", 1000.0), ("sustained", None)]
LNAME = {"100": "p99 ≤ 100 ms", "200": "p99 ≤ 200 ms", "500": "p99 ≤ 500 ms", "1000": "p99 ≤ 1 s",
         "sustained": "sustained (achieved ≥ 97%, p99 ≤ 2 s)"}
LCOL = {"100": "#c9372c", "200": "#eb6834", "500": "#eda100", "1000": "#1baf7a", "sustained": "#2a78d6"}


def load(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(f"{d}/*.jsonl"):
            if "smoke" in f:
                continue
            rows += [json.loads(x) for x in open(f) if x.strip()]
    # the fine P5s grid supersedes the coarse one for the same (config, strategy, L)
    fine = {(r["config"], r["strategy"], r["L"]) for r in rows if r.get("tag") == "p5s-fine"}
    rows = [r for r in rows if r.get("tag") != "p5s" or (r["config"], r["strategy"], r["L"]) not in fine]
    return [r for r in rows if not r.get("cotenant")]


def family(r, level):
    """Which curve a row belongs to at this level (None = not used at this level)."""
    strict = level == "100"
    if (r.get("tag") == "p5x") == strict:
        return None
    if r["kind"] == "recompute":
        return "recompute"
    if strict:
        return "fetch" if r["strategy"] == "verbs" else None
    return {"raw.t.fused": "fetch", "verbs.b.5a": "fetch_basic", "verbs.t.fused": "fetch_verbs_tuned"}.get(r["strategy"])


def passes(r, level):
    if r.get("err_requests", 0) or r.get("bad_words", 0):
        return False
    if level == "sustained":
        return r["achieved"] >= 0.97 * r["offered"] and r["p99_ms"] <= 2000.0
    return r["p99_ms"] <= float(level) and r["achieved"] >= 0.9 * r["offered"]


def capacity(rows, level):
    """Highest offered load that passes; at 'sustained', the rate actually delivered there (min(offered, achieved)),
    so a point slightly past the ceiling with achieved >= 97% does not overstate the capacity."""
    ok = [min(r["offered"], r["achieved"]) if level == "sustained" else r["offered"] for r in rows if passes(r, level)]
    return max(ok) if ok else None


def breakeven(pts):
    """pts: [(L, ratio fetch/recompute)] -> L where the ratio crosses 1 (last crossing, log-log interpolation)."""
    be = None
    for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
        if r0 < 1 <= r1:
            be = math.exp(math.log(l0) + (0 - math.log(r0)) * (math.log(l1) - math.log(l0)) / (math.log(r1) - math.log(r0)))
    return be


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = load(a.dirs)
    cap = {}
    keys = set()
    for lv, _ in LEVELS:
        by = defaultdict(list)
        for r in rows:
            fam = family(r, lv)
            if fam:
                by[(r["config"], fam, r["L"])].append(r)
        for k, v in by.items():
            cap[(k, lv)] = capacity(v, lv)
            keys.add(k)
    by = {k: None for k in keys}
    cfgs = [c for c in ("S", "M", "L") if any(k[0] == c for k in by)]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.4 * len(cfgs), 3.8), dpi=140, facecolor=SURFACE, squeeze=False)
    summary = {}
    for ax, c in zip(axes[0], cfgs):
        ax.set_facecolor(SURFACE)
        Ls = sorted({k[2] for k in by if k[0] == c and k[1] == "recompute"} & {k[2] for k in by if k[0] == c and k[1] == "fetch"})
        for lv, _ in LEVELS:
            pts = []
            for L in Ls:
                f, rc = cap.get(((c, "fetch", L), lv)), cap.get(((c, "recompute", L), lv))
                if f and rc:
                    pts.append((L, f / rc))
            if not pts:
                continue
            summary[(c, lv)] = (pts, breakeven(pts))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=LCOL[lv], marker="o", ms=4.5, lw=1.7,
                    mec=SURFACE, mew=1, label=LNAME[lv], zorder=3)
        ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(Ls)
        ax.set_xticklabels([f"{L / 1024:g}k" for L in Ls], fontsize=8, color=INK2)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["¼", "½", "1", "2", "4"], fontsize=8, color=INK2)
        ax.set_ylim(0.3, 4.5)
        ax.set_xlabel("history length (tokens)", color=INK2, fontsize=8.5)
        ax.set_title(f"config {c}", color=INK, fontsize=10, loc="left")
        ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("fetch ÷ recompute capacity (users/s, 1 GPU)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.12))
    fig.suptitle("Serving capacity of remote fetch relative to recompute, per latency target (above 1: fetch serves more)",
                 color=INK, fontsize=10, y=1.17, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.md:
        for c in cfgs:
            Ls = sorted({k[2] for k in by if k[0] == c and k[1] == "recompute"})
            print(f"\n**Config {c}: capacity (users/s, 1 GPU) recompute / fetch [basic CPU verbs, CPU verbs + fused wait]**\n")
            print("| Target | " + " | ".join(f"{L / 1024:g}k" for L in Ls) + " | break-even |")
            print("|---|" + "---|" * (len(Ls) + 1))
            for lv, _ in LEVELS:
                cells = []
                for L in Ls:
                    rc, f = cap.get(((c, "recompute", L), lv)), cap.get(((c, "fetch", L), lv))
                    fb, ft = cap.get(((c, "fetch_basic", L), lv)), cap.get(((c, "fetch_verbs_tuned", L), lv))
                    fmt = lambda x: f"{x:,.0f}" if x else "–"  # noqa: E731
                    cells.append(f"{fmt(rc)} / {fmt(f)} [{fmt(fb)}, {fmt(ft)}]")
                be = summary.get((c, lv), (None, None))[1]
                print(f"| {LNAME[lv]} | " + " | ".join(cells) + f" | {be / 1024:.1f}k |" if be else
                      f"| {LNAME[lv]} | " + " | ".join(cells) + " | – |")
        n_chk = sum(r.get("checks", 0) for r in rows if r.get("tag") == "p5x")
        n_bad = sum(r.get("bad_words", 0) for r in rows if r.get("tag") == "p5x")
        print(f"\nP5x data checks: {n_chk} sampled batches, {n_bad} mismatched words")


if __name__ == "__main__":
    main()
