#!/usr/bin/env python3
"""Explanatory diagrams of docs/results.md (schematic, not measured):
  fetch-timeline.png  one request's transfer and compute over time with each wait variant (basic wait kernel, fused
                      wait, per-tile wait), for a long history (transfer-bound) and a short one (compute-bound)
  fetch-who.png       who posts the reads and who reports progress, CPU verbs vs raw GDAKI, per wait variant
    tools/fetch_diagrams.py docs/figs"""
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# kind: (fill, edge, text, hatch)
STY = {
    "nic": ("#9FE1CB", "#0F6E56", "#04342C", None),
    "attn": ("#AFA9EC", "#534AB7", "#26215C", None),
    "spin": ("#EEEDFE", "#7F77DD", "#3C3489", "////"),
    "waitk": (SURFACE, "#5F5E5A", "#2C2C2A", "////"),
    "other": ("#D3D1C7", "#888780", "#2C2C2A", None),
}
LANES = ["NIC reads", "wait kernel (basic)", "fused wait", "per-tile wait"]


def timeline(ax, nic, lanes, title, xmax):
    """nic: [(x0, x1, label)] one block per layer (drawn as 5 chunks); lanes: 3 lists of (x0, x1, kind, label)."""
    ax.set_facecolor(SURFACE)
    h = 0.62
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    charw = 0.058 * xmax / width_in                          # x units per character at 7.5 pt
    ys = [3, 2, 1, 0]
    for x0, x1, lab in nic:
        n = 5
        for k in range(n):
            a, b = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
            fc, ec, tc, _ = STY["nic"]
            ax.add_patch(Rectangle((a, ys[0] - h / 2), b - a, h, facecolor=fc, edgecolor=ec, lw=0.6))
        ax.text((x0 + x1) / 2, ys[0], lab, ha="center", va="center", fontsize=7.5, color=STY["nic"][2])
    for y, segs in zip(ys[1:], lanes):
        end = 0
        for x0, x1, kind, lab in segs:
            fc, ec, tc, hatch = STY[kind]
            ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, facecolor=fc, edgecolor=ec, lw=0.6, hatch=hatch))
            if lab and (x1 - x0) >= (len(lab) + 1.5) * charw:
                ax.text((x0 + x1) / 2, y, lab, ha="center", va="center", fontsize=7.5, color=tc,
                        bbox=dict(facecolor=fc, edgecolor="none", pad=0.6) if hatch else None)
            end = max(end, x1)
        ax.plot([end, end], [y - 0.42, y + 0.42], color=INK, lw=1.1)
        ax.text(end + 0.35 * charw * 2, y, "done", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.3, xmax)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(LANES, fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_xlabel("time →  (schematic, not to scale)", fontsize=7.5, color=MUTED, loc="right")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def fig_timeline(out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), dpi=140, facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.62, left=0.14, right=0.98, top=0.86, bottom=0.08))
    # (a) long history, transfer-bound: every layer's data takes longer to arrive than the layer's compute
    nic = [(0, 10, "layer 0"), (10, 20, "layer 1"), (20, 30, "layer 2")]
    g = 0.6   # launch of the attention after the wait kernel ends
    basic = [(0, 1, "other", ""), (1, 10, "waitk", "wait kernel"), (10 + g, 13 + g, "attn", "attn 0"),
             (13 + g, 15 + g, "other", ""), (15 + g, 20, "waitk", "wait"), (20 + g, 23 + g, "attn", "attn 1"),
             (23 + g, 25 + g, "other", ""), (25 + g, 30, "waitk", "wait"), (30 + g, 33 + g, "attn", "attn 2"),
             (33 + g, 35 + g, "other", "")]
    fused = [(0, 1, "other", ""), (1, 10, "spin", "attention spins"), (10, 13, "attn", "attn 0"), (13, 15, "other", ""),
             (15, 20, "spin", "spins"), (20, 23, "attn", "attn 1"), (23, 25, "other", ""), (25, 30, "spin", "spins"),
             (30, 33, "attn", "attn 2"), (33, 35, "other", "")]
    tile = [(0, 1, "other", ""), (1, 10.5, "attn", "attn 0, as chunks land"), (10.5, 12.5, "other", ""),
            (12.5, 20.5, "attn", "attn 1, as chunks land"), (20.5, 22.5, "other", ""),
            (22.5, 30.5, "attn", "attn 2, as chunks land"), (30.5, 32.5, "other", "")]
    timeline(axes[0], nic, [basic, fused, tile], "(a) Long history: the transfer dominates. Per-tile waits end about "
             "one layer's attention earlier", 38.5)
    # (b) short history, compute-bound: the data is there before the compute needs it
    nic = [(0, 2, "L0"), (2, 4, "L1"), (4, 6, "L2")]
    w, g = 0.7, 0.5
    basic, t = [(0, 2, "other", "")], 2
    for i in range(3):
        basic += [(t, t + w, "waitk", ""), (t + w + g, t + w + g + 3, "attn", f"attn {i}")]
        t = t + w + g + 3
        basic += [(t, t + (4 if i < 2 else 2.5), "other", "")]
        t += 4 if i < 2 else 2.5
    fused, t = [(0, 2, "other", "")], 2
    for i in range(3):
        fused += [(t, t + 3, "attn", f"attn {i}"), (t + 3, t + 3 + (4 if i < 2 else 2.5), "other", "")]
        t += 3 + (4 if i < 2 else 2.5)
    timeline(axes[1], nic, [basic, fused, list(fused)], "(b) Short history: the compute dominates. The fused wait "
             "saves a wait kernel and a launch per layer; per-tile waits add nothing", 27)
    handles = [Patch(facecolor=STY["nic"][0], edgecolor=STY["nic"][1], label="NIC read of a chunk"),
               Patch(facecolor=STY["attn"][0], edgecolor=STY["attn"][1], label="attention computing"),
               Patch(facecolor=STY["spin"][0], edgecolor=STY["spin"][1], hatch="////",
                     label="attention kernel resident, spinning until its layer has landed"),
               Patch(facecolor=STY["waitk"][0], edgecolor=STY["waitk"][1], hatch="////",
                     label="1-thread wait kernel"),
               Patch(facecolor=STY["other"][0], edgecolor=STY["other"][1], label="projections, layer output, head")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK,
               bbox_to_anchor=(0.01, 0.995), handlelength=1.6, columnspacing=1.4)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


# ---------------------------------------------------------------- who posts, who reports progress
BOX = {  # fill, edge, title, body
    "cpu": ("#FAECE7", "#D85A30", "#712B13", "#993C1D"),
    "gpu": ("#EEEDFE", "#7F77DD", "#3C3489", "#534AB7"),
    "nic": ("#E1F5EE", "#1D9E75", "#085041", "#0F6E56"),
    "mem": ("#F1EFE8", "#888780", "#444441", "#5F5E5A"),
}


def box(ax, x, y, w, h, kind, title, body, wrap):
    fc, ec, tc, bc = BOX[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", facecolor=fc, edgecolor=ec,
                                lw=0.7))
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=8.5, color=tc, fontweight="bold")
    ax.text(x + 0.12, y + h - 0.33, "\n".join(textwrap.wrap(body, wrap, break_on_hyphens=False)), ha="left", va="top",
            fontsize=7.8,
            color=bc, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.8,
                                 shrinkA=1, shrinkB=1))


def fig_who(out):
    W, H = 10.0, 7.3
    fig = plt.figure(figsize=(W, H), dpi=140, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    bw, bh, pitch, wrap = 4.55, 0.82, 1.05, 76
    cols = {"verbs": 0.3, "raw": 5.15}
    ax.text(cols["verbs"] + bw / 2, H - 0.25, "CPU verbs", ha="center", va="center", fontsize=10, color=INK,
            fontweight="bold")
    ax.text(cols["raw"] + bw / 2, H - 0.25, "raw GDAKI", ha="center", va="center", fontsize=10, color=INK,
            fontweight="bold")
    verbs = [
        ("cpu", "Request thread (CPU)", "Posts every read, then launches the GPU work. Basic and fused: one read per "
         "user's K or V block; per-tile: 128-row chunks from the history's end, round-robin over 4 queue pairs."),
        ("nic", "NIC", "Reads the remote host memory into GPU memory; completions land in host memory."),
        ("cpu", "Worker thread (CPU, spinning)", "Polls the completion queue and counts what has landed, per layer "
         "(per-tile: per chunk step)."),
        ("mem", "Progress words in host memory", "Per layer: 'layer landed' (basic, fused) or 'chunk steps landed' "
         "(per-tile)."),
        ("gpu", "Relay kernel (GPU, 1 thread)", "Fused and per-tile: copies the words into GPU memory for the "
         "attention kernel. Basic: no relay; the wait kernel reads host memory over PCIe."),
    ]
    raw = [
        ("cpu", "Request thread (CPU)", "Writes the request's index into pinned host memory and launches the GPU "
         "work. Posts nothing and polls nothing."),
        ("gpu", "Post kernel (GPU, side branch of the graph)", "Writes the reads and rings the NIC doorbell. Basic and "
         "fused: one read per K or V block; per-tile: the chunks, one batch per layer and queue pair, 32 lanes."),
        ("nic", "NIC", "Reads the remote host memory into GPU memory; completions land in GPU memory."),
        ("gpu", "Completion polling (GPU)", "Basic: the wait kernel itself. Fused: a 1-thread poller kernel. "
         "Per-tile: the post kernel, once everything is posted."),
        ("mem", "Progress words in GPU memory", "Per layer: 'layer landed' (fused) or 'chunk steps landed' "
         "(per-tile). Basic: none."),
    ]
    top = H - 0.55
    for name, rows in (("verbs", verbs), ("raw", raw)):
        x = cols[name]
        for i, (kind, title, body) in enumerate(rows):
            y = top - bh - i * pitch
            box(ax, x, y, bw, bh, kind, title, body, wrap)
            if i:
                arrow(ax, x + bw / 2, y + bh + (pitch - bh) - 0.02, x + bw / 2, y + bh + 0.02)
    by = 0.45
    box(ax, 0.3, by, 9.4, 0.9, "gpu", "Who waits before using the data (both mechanisms)",
        "Basic: a 1-thread wait kernel before each layer's attention. Fused: the attention kernel itself, before its "
        "first K/V load, for the whole layer. Per-tile: the attention kernel, before each K tile, only for the chunks "
        "that tile needs (the history arrives from its end, the order the attention reads it).", 178)
    last = top - bh - 4 * pitch
    for name in ("verbs", "raw"):
        x = cols[name] + bw / 2
        arrow(ax, x, last - 0.02, x, by + 0.9 + 0.02)
    handles = [Patch(facecolor=BOX[k][0], edgecolor=BOX[k][1], label=lab)
               for k, lab in (("cpu", "CPU"), ("gpu", "GPU"), ("nic", "NIC"), ("mem", "memory word"))]
    ax.legend(handles=handles, loc="lower right", ncol=4, frameon=False, fontsize=7.5, labelcolor=INK,
              bbox_to_anchor=(0.985, 0.0), handlelength=1.4, columnspacing=1.2)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "docs/figs"
    fig_timeline(f"{d}/fetch-timeline.png")
    fig_who(f"{d}/fetch-who.png")
