#!/usr/bin/env python3
"""P6s report: the fetch server next to a busy GPU (results/p6s).
    tools/p6s_report.py results/p6s -o docs/figs/p6s-contention.png [--md]
Conditions per transport: alone; sharing the GPU with a co-located recompute job; the co-located job confined to a
green context without 8 SMs (GDAKI posts on the 8-SM one)."""
import argparse
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TCOL = {"verbs": "#2a78d6", "raw": "#1baf7a"}
TNAME = {"verbs": "CPU verbs", "raw": "raw GDAKI"}
COND = {"alone": ("-", "o", "alone"), "shared": ((0, (5, 2)), "s", "GPU shared with a recompute job"),
        "prio": ((0, (3, 1, 1, 1)), "D", "shared, server on high-priority streams"),
        "green": ((0, (1, 1.5)), "^", "shared, job confined to 124 SMs (green context)")}


def cond(r):
    """Green rows count only when the co-located job really ran in its partition: its graph must be captured on the
    green-context stream (runs before that fix replayed a primary-context graph, which used every SM: its batch time
    was the unconfined 13.3 ms instead of 13.8-14.0 ms on 124 SMs)."""
    if not r.get("cotenant"):
        return "alone"
    if r.get("green"):
        return "green" if (r.get("cotenant_ms") or 0) > 13.55 else None
    return "prio" if r.get("priority") else "shared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in glob.glob(f"{a.dir}/*.jsonl"):
        rows += [json.loads(x) for x in open(f) if x.strip()]
    by = defaultdict(list)
    for r in rows:
        if cond(r) is not None:
            by[(r["L"], r["strategy"], cond(r))].append(r)
    Ls = sorted({r["L"] for r in rows})
    trs = [t for t in ("verbs", "raw") if any(r["strategy"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(Ls), figsize=(5.0 * len(Ls), 3.7), dpi=140, facecolor=SURFACE, squeeze=False)
    cap = 300.0
    for ax, L in zip(axes[0], Ls):
        ax.set_facecolor(SURFACE)
        for t in trs:
            for c, (ls, mk, cname) in COND.items():
                pts = sorted(by.get((L, t, c), []), key=lambda r: r["frac"])
                if not pts:
                    continue
                ax.plot([100 * r["frac"] for r in pts], [min(r["p99_ms"], cap) for r in pts], color=TCOL[t], ls=ls,
                        marker=mk, ms=5, lw=1.7, mec=SURFACE, mew=1, label=f"{TNAME[t]}, {cname}", zorder=3)
        ax.axhline(100, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_ylim(0, cap * 1.04)
        ax.set_title(f"config M, history {L // 1024}k", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("offered load (% of the server's unshared ceiling)", color=INK2, fontsize=8.5)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("p99 latency of the fetch server (ms, capped at 300)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.16))
    fig.suptitle("Remote fetch next to a busy GPU: a co-located recompute job (M, 4k, batches of 8) runs back to back",
                 color=INK, fontsize=10, y=1.21, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.md:
        print("\n**p50 / p99 latency (ms) and share within the SLO; co-located job's throughput (% of alone)**\n")
        print("| History | Transport | Condition | 50% load | 75% load | 90% load | co-located job |")
        print("|---|---|---|---|---|---|---|")
        for L in Ls:
            for t in trs:
                for c in COND:
                    pts = sorted(by.get((L, t, c), []), key=lambda r: r["frac"])
                    if not pts:
                        continue
                    cells = [f"{r['p50_ms']:.1f} / {r['p99_ms']:.0f} ({100 * r['success']:.0f}%)" for r in pts]
                    cot = [r["cotenant_rate"] / r["cotenant_alone"] for r in pts if r.get("cotenant_rate")]
                    cells += [""] * (3 - len(cells))
                    print(f"| {L // 1024}k | {TNAME[t]} | {COND[c][2]} | " + " | ".join(cells) + " | "
                          + (f"{100 * min(cot):.0f}–{100 * max(cot):.0f}%" if cot else "–") + " |")


if __name__ == "__main__":
    main()
