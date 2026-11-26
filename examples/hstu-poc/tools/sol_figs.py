#!/usr/bin/env python3
"""Figures of the speed-of-light analysis (docs/sol.md). In the container (matplotlib):
    python tools/sol_figs.py docs/figs
  sol-timeline.png  one request over time: data landed per layer (NIC) and each layer's attention (GPU), for the
                    local hit and the two fastest RDMA implementations, against the SoL end; M, one user: 0.5k history
                    (compute-bound) and 4k (transfer-bound: its first and last 150 us)
  sol-ratio.png     measured latency / SoL per history length: recompute, local hit (compute only), fastest fetch
  sol-gaps.png      fetch latency split into start (issue -> first chunk step landed), transfer (-> last landed) and
                    tail (-> output seen on the host), next to the same split at SoL (B = 1)
Data: results/tl4 (local, recompute; verbs.t2.tile for sol-gaps), results/tl6 (one session: CPU verbs per-tile v2 and raw GDAKI
per-tile v3, the final implementations), results/p2t
(full grid for sol-ratio)."""
import glob
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sol  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
C_LOCAL, C_VERBS, C_RAW, C_REC, C_SOL, C_DATA = "#6b7785", "#1f6fb2", "#c25700", "#7a3fa0", "#2a9d5c", "#b8c9db"
FULL = 0x7fffffff


def tl_rows(path, mode, c, B, L):
    rows = []
    for f in glob.glob(path):
        for x in open(f):
            r = json.loads(x)
            if r["mode"] == mode and r["config"] == c and int(r["B"]) == B and int(r["L"]) == L:
                rows.append(r)
    return rows


def median_request(rows):
    rows = sorted(rows, key=lambda r: r["host"][3])
    return rows[len(rows) // 2]


def data_events(r):
    ev = r["cpu"]["ev"] if "cpu" in r and r["cpu"]["ev"] else r.get("pub_ev", [])
    return [(l, v, t) for l, v, t in ev if t is not None]


def draw_request(ax, r, y, color, label, xlim, data=True):
    """One implementation: attention spans per layer (bar) and the history landed per layer (thin bars below)."""
    n = len(r["attn"])
    for l in range(n):
        a, b = r["attn"][l][0], r["attn"][l][1]
        if a is None or b is None:
            continue
        ax.barh(y, (b - a) / 1e3, left=a / 1e3, height=0.36, color=color, alpha=0.9, lw=0)
        lo, hi = max(a / 1e3, xlim[0]), min(b / 1e3, xlim[1])
        if hi - lo > 0.04 * (xlim[1] - xlim[0]):     # label the visible part of the bar
            ax.text((lo + hi) / 2, y, f"layer {l}" if hi - lo > 0.3 * (xlim[1] - xlim[0]) else str(l), ha="center",
                    va="center", fontsize=6.5, color="white", clip_on=True)
    if data:
        ev = data_events(r)
        for l in range(n):
            ts = [t for ll, v, t in ev if ll == l]
            if ts:
                t0 = min(ts) if l == 0 else max(t for ll, v, t in ev if ll == l - 1)
                ax.barh(y - 0.36, (max(ts) - t0) / 1e3, left=t0 / 1e3, height=0.16, color=C_DATA, lw=0.5,
                        edgecolor="#7d93ab")
    ax.plot([r["host"][3] / 1e3], [y], marker="|", ms=14, mew=2.2, color="black")


def fig_timeline(out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    cases = [("M", 1, 512, None, "M, one user, 0.5k history: compute-bound"),
             ("M", 1, 4096, (0, 150), "M, one user, 4k: transfer-bound, first 150 µs"),
             ("M", 1, 4096, "end", "M, one user, 4k: last 150 µs")]
    for ax, (c, B, L, win, title) in zip(axes, cases):
        local = tl_rows(os.path.join(RES, "tl4", "recompute.timeline.rank0.jsonl"), "local", c, B, L)
        verbs = tl_rows(os.path.join(RES, "tl6", "verbs.t2.tile.timeline.rank0.jsonl"), "verbs.t2.tile", c, B, L)
        raw = tl_rows(os.path.join(RES, "tl6", "raw.pl.tile.timeline.rank0.jsonl"), "raw.pl.tile", c, B, L)
        rows = [(median_request(local), C_LOCAL, "local hit (compute only)", False),
                (median_request(verbs), C_VERBS, "CPU verbs, per-tile v2", True),
                (median_request(raw), C_RAW, "raw GDAKI, per-tile v3", True)]
        end = max(r["host"][3] for r, *_ in rows) / 1e3
        xlim = (end - 150, end + 5) if win == "end" else (win if win else (0, end * 1.03))
        for i, (r, col, lab, data) in enumerate(rows):
            draw_request(ax, r, 2 - i, col, lab, xlim, data=data)
        sl, parts = sol.sol_fetch(c, B, L)
        ax.axvline(sl * 1e6, color=C_SOL, ls="--", lw=1.4)
        ax.axvline(parts["compute"] * 1e6, color=C_LOCAL, ls=":", lw=1.2)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([lab for _, _, lab, _ in rows] if ax is axes[0] else [])
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.75, 2.5)
        ax.set_xlabel("µs since the request was issued (host)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].text(0.99, 0.02, "dashed green: fetch SoL;  dotted grey: compute SoL", transform=axes[0].transAxes,
                 ha="right", fontsize=7.5, color="#333")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color=C_LOCAL, label="attention, local hit"), Patch(color=C_VERBS, label="attention, CPU verbs"),
                        Patch(color=C_RAW, label="attention, raw GDAKI"),
                        Patch(facecolor=C_DATA, edgecolor="#7d93ab", label="history of layer l landing"),
                        Line2D([], [], color="black", marker="|", ls="", ms=10, mew=2, label="output seen (host)"),
                        Line2D([], [], color=C_SOL, ls="--", label="fetch SoL")],
               loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, "sol-timeline.png"), dpi=130)
    plt.close(fig)


def fig_ratio(out):
    meas = sol.load_measured(os.path.join(RES, "p2t"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, c in zip(axes, ("S", "M", "L")):
        for B, ls in ((1, "-"), (8, "--")):
            pts = sorted((L, m) for (cc, BB, L), m in meas.items() if cc == c and BB == B)
            Ls = [L for L, _ in pts]
            rec = [m["recompute"] / (sol.sol_recompute(c, B, L) * 1e3) if "recompute" in m else None for L, m in pts]
            loc = [m["local"] / (sol.sol_fetch(c, B, L)[1]["compute"] * 1e3) if "local" in m else None for L, m in pts]
            fet = [sol.best_fetch(m)[1] / (sol.sol_fetch(c, B, L)[0] * 1e3) if sol.best_fetch(m)[1] else None
                   for L, m in pts]
            for ys, col, lab in ((rec, C_REC, "recompute"), (loc, C_LOCAL, "local hit (compute only)"),
                                 (fet, C_VERBS, "fetch, fastest implementation")):
                xy = [(L, y) for L, y in zip(Ls, ys) if y]
                ax.plot([L for L, _ in xy], [y for _, y in xy], ls=ls, color=col, marker="o", ms=3,
                        label=f"{lab}, B = {B}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhline(1, color=C_SOL, lw=1)
        ax.set_title(f"config {c}", fontsize=10)
        ax.set_xlabel("history tokens per user")
        ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384, 32768])
        ax.set_xticklabels(["0.5k", "1k", "2k", "4k", "8k", "16k", "32k"])
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("measured latency ÷ speed of light")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, fontsize=8, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "sol-ratio.png"), dpi=130)
    plt.close(fig)


def fig_gaps(out):
    cells = [(c, 1, L) for c in ("S", "M", "L") for L in (512, 1024, 4096, 16384)]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = 0
    ticks, labels = [], []
    for c, B, L in cells[::-1]:
        rows = tl_rows(os.path.join(RES, "tl4", "verbs.t2.tile.timeline.rank0.jsonl"), "verbs.t2.tile", c, B, L)
        if not rows:
            continue
        first = statistics.median(min(t for l, v, t in data_events(r) if l == 0) for r in rows) / 1e3
        last = statistics.median(max(t for l, v, t in data_events(r)) for r in rows) / 1e3
        seen = statistics.median(r["host"][3] for r in rows) / 1e3
        n, h, H, d = sol.CFG[c]
        rows_t = rows[0].get("tile_rows") or 128
        step = B * 2 * min(rows_t, L) * H * d * 2
        s_first = (sol.LAT["t_rdma"] + step / sol.HW["nic"]) * 1e6
        s_last = (sol.LAT["t_rdma"] + sol.kv_bytes(c, B, L) / sol.HW["nic"]) * 1e6
        s_end = sol.sol_fetch(c, B, L)[0] * 1e6
        for yy, (a, b, e), alpha in ((y + 0.22, (first, last, seen), 1.0), (y - 0.22, (s_first, s_last, s_end), 0.45)):
            tot = s_end                                  # both bars in units of the SoL latency
            ax.barh(yy, a / tot, color="#e0a458", height=0.38, alpha=alpha, lw=0)
            ax.barh(yy, (b - a) / tot, left=a / tot, color="#5b8db8", height=0.38, alpha=alpha, lw=0)
            ax.barh(yy, (e - b) / tot, left=b / tot, color="#8e6bb0", height=0.38, alpha=alpha, lw=0)
            ax.text(e / tot + 0.02, yy, f"{e:,.0f} µs" + (f"  ({e / s_end:.2f}× SoL)" if alpha == 1.0 else "  (SoL)"),
                    va="center", fontsize=7.5)
        ticks.append(y)
        labels.append(f"{c}, {L / 1024:g}k")
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 3.6)
    ax.axvline(1, color=C_SOL, lw=1, ls="--")
    ax.set_xlabel("time ÷ the request's SoL latency (upper bar: measured, CPU verbs per-tile v2; lower bar: SoL)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e0a458", label="start: issue → first chunk step landed"),
                       Patch(color="#5b8db8", label="transfer: → last byte landed"),
                       Patch(color="#8e6bb0", label="tail: → output seen by the host")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5, frameon=False)
    ax.set_title("one user per request: the excess over SoL is in the tail (compute after the last byte) and, for "
                 "compute-bound requests, in the compute itself", fontsize=9, pad=24)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sol-gaps.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    o = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "figs")
    os.makedirs(o, exist_ok=True)
    fig_timeline(o)
    fig_ratio(o)
    fig_gaps(o)
    print("wrote sol-timeline.png, sol-ratio.png, sol-gaps.png in", o)
