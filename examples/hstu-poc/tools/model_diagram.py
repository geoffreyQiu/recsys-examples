#!/usr/bin/env python3
"""Block diagram of the HSTU ranking model as run by the POC (hstu_poc/model.py): request, pre-processing, HSTU layer,
post-processing, and where the remote-fetch path differs from recompute.
    python tools/model_diagram.py docs/figs        (in the container: needs matplotlib)"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, SUB, BOX, OPF, ORANGE, KVF = "#1b2733", "#4a5a6c", "#f3f5f8", "#ffffff", "#c25700", "#fff2e3"
W, H = 1080, 734


def fig_ax():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc=BOX, ec="#3a4a5e", lw=1.1, r=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def text(ax, x, y, s, size=9.4, color=INK, weight="normal", style="normal", ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style, ha=ha, va="center")


def arrow(ax, pts, color="#333333", lw=1.1, ls="-"):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=10, shrinkA=0, shrinkB=0))


def main(out):
    fig, ax = fig_ax()
    M, S = 8.6, 7.7                                   # main and secondary font sizes
    # ---------------- request
    box(ax, 12, 30, 262, 178)
    text(ax, 24, 50, "Request: B users (jagged batch)", 9.8, weight="bold")
    text(ax, 24, 72, "user u, history of L_u tokens:", M)
    text(ax, 36, 89, "item ids i_1 … i_L", S, SUB)
    text(ax, 36, 104, "action ids a_1 … a_L", S, SUB)
    text(ax, 36, 119, "positions 0 … L_u − 1", S, SUB)
    text(ax, 24, 141, "C = 256 candidates to rank:", M)
    text(ax, 36, 158, "item ids c_1 … c_C (no action)", S, SUB)
    text(ax, 24, 180, "users concatenated, no padding", S, SUB)
    text(ax, 24, 195, "(offsets: cu_seqlens)", S, SUB)
    # ---------------- pre-processing
    box(ax, 12, 240, 262, 270)
    text(ax, 24, 260, "Pre-processing (embedding)", 9.8, weight="bold")
    ops = [("E_item[i] + E_act[a]", "actions: history tokens only"), ("· W_in   (128 → h)", None),
           ("+ E_pos[bucket(position)]", None), ("LayerNorm", None)]
    y = 274
    for i, (a, b) in enumerate(ops):
        hh = 40 if b else 26
        box(ax, 27, y, 232, hh, fc=OPF, lw=0.9, r=3)
        text(ax, 37, y + 13, a, M)
        if b:
            text(ax, 37, y + 29, b, S, SUB)
        if i:
            arrow(ax, [(143, y - 12), (143, y)])
        y += hh + 12
    text(ax, 27, y + 8, "x : T × h, where", M)
    text(ax, 27, y + 26, "recompute: T = Σ_u (L_u + C)", S, SUB)
    text(ax, 27, y + 41, "fetch: T = B · C (candidates only)", S, SUB)
    arrow(ax, [(143, 208), (143, 238)])
    # ---------------- HSTU layer
    X0, X1 = 300, 745                                 # layer box
    OX, OW = 345, 360                                 # op boxes
    box(ax, X0, 30, X1 - X0, 562, r=8)
    text(ax, X0 + 14, 50, "HSTU layer l   (× n layers)", 9.8, weight="bold")
    text(ax, X0 + 230, 50, "H heads × head dim d,  hd = H·d", S, SUB)
    arrow(ax, [(274, 455), (288, 455), (288, 82), (OX, 82)])
    text(ax, 280, 70, "x_l", M, style="italic")
    box(ax, OX, 68, OW, 28, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 82, "LayerNorm", M)
    cx = OX + OW / 2
    arrow(ax, [(cx, 96), (cx, 112)])
    box(ax, OX, 112, OW, 28, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 126, "Linear h → 4·hd, + bias, SiLU    (one GEMM)", M)
    xs = [OX + 10 + i * 90 for i in range(4)]
    for x0, lab in zip(xs, ("U", "V", "Q", "K")):
        arrow(ax, [(x0 + 30, 140), (x0 + 30, 160)])
        box(ax, x0, 160, 60, 26, fc=OPF, lw=0.9, r=3)
        text(ax, x0 + 30, 173, lab, 10, style="italic", ha="center")
    text(ax, xs[3] + 66, 173, "T × hd", S, SUB)
    ux, vx, qx, kx = (x + 30 for x in xs)
    box(ax, vx - 30, 222, kx + 40 - (vx - 30), 64, fc=KVF, ec=ORANGE, lw=1.3, r=4)
    text(ax, vx - 20, 238, "KV buffer of layer l (GPU memory)", M, weight="bold")
    text(ax, vx - 20, 256, "K, V: [tokens × H × d] per user:", S, SUB)
    text(ax, vx - 20, 272, "history rows | candidates' rows", S, SUB)
    arrow(ax, [(vx, 186), (vx, 222)])
    arrow(ax, [(kx, 186), (kx, 222)])
    box(ax, OX, 308, OW, 54, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 323, "Attention   A = ( SiLU(α · Q Kᵀ) / N  ⊙  M ) · V", M)
    text(ax, OX + 12, 340, "no softmax;  M: causal on the history,", S, SUB)
    text(ax, OX + 12, 354, "each candidate sees the history and itself", S, SUB)
    arrow(ax, [(qx, 186), (qx, 204), (OX + 22, 204), (OX + 22, 308)])
    arrow(ax, [(qx + 45, 286), (qx + 45, 308)])
    text(ax, qx + 51, 297, "K, V", S, SUB)
    arrow(ax, [(cx, 362), (cx, 378)])
    box(ax, OX, 378, OW, 28, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 392, "U ⊙ LayerNorm(A)", M)
    arrow(ax, [(ux, 186), (ux, 196), (330, 196), (330, 392), (OX, 392)])
    arrow(ax, [(cx, 406), (cx, 422)])
    box(ax, OX, 422, OW, 28, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 436, "Linear hd → h, + bias    (one GEMM)", M)
    arrow(ax, [(cx, 450), (cx, 466)])
    box(ax, OX, 466, OW, 28, fc=OPF, lw=0.9, r=3)
    text(ax, OX + 12, 480, "+ residual x_l   →   x_(l+1)", M)
    arrow(ax, [(314, 82), (314, 480), (OX, 480)], ls="--")
    text(ax, OX, 512, "per layer: 2 GEMMs, 1 attention, 3 LayerNorms,", S, SUB)
    text(ax, OX, 527, "element-wise ops (POC: ~12 kernels per layer)", S, SUB)
    ax.plot([OX + OW, 725, 725, X1 + 2], [480, 480, 572, 572], color="#333333", lw=1.1)
    text(ax, 600, 584, "after the last layer", S, SUB)
    # ---------------- remote store and RDMA
    RX = 775
    box(ax, RX, 120, 293, 104, fc=KVF, ec=ORANGE, lw=1.3)
    text(ax, RX + 12, 140, "Remote KV store", 9.8, weight="bold")
    for i, s_ in enumerate(("host memory of another node (no GPU)", "per user: [layer][K|V][L][H][d]",
                            "written by earlier requests", "(always a hit in this study)")):
        text(ax, RX + 12, 160 + 15 * i, s_, S, SUB)
    arrow(ax, [(RX, 226), (752, 250), (kx + 42, 250)], color=ORANGE, lw=1.6)
    for i, s_ in enumerate(("RDMA reads over one 400 Gb/s NIC:", "all layers posted at request start,",
                            "layer by layer, 128-row chunks", "from the history's end")):
        text(ax, RX + 12, 268 + 15 * i, s_, S + 0.3, ORANGE)
    arrow(ax, [(RX + 6, 340), (OX + OW + 2, 340)], color=ORANGE, lw=1.4, ls="--")
    text(ax, RX + 12, 340, "waits: per layer or per K tile", S + 0.3, ORANGE)
    text(ax, RX + 12, 355, "(attention reads a tile once landed)", S + 0.3, ORANGE)
    text(ax, RX + 12, 382, "recompute instead runs every history", S, SUB)
    text(ax, RX + 12, 397, "token through all layers", S, SUB)
    # ---------------- post-processing
    box(ax, RX, 436, 293, 156)
    text(ax, RX + 12, 456, "Post-processing (head)", 9.8, weight="bold")
    for i, s_ in enumerate(("x[candidate rows]: B·C × h", "LayerNorm", "MLP h → 512 (ReLU) → 8", "sigmoid")):
        text(ax, RX + 12, 480 + 20 * i, s_, M)
    text(ax, RX + 12, 566, "scores: B·C × 8 tasks", M, weight="bold")
    arrow(ax, [(X1 + 2, 572), (RX - 2, 530)])
    # ---------------- configs
    box(ax, 12, 604, 1056, 122, fc="#ffffff", ec="#9aa7b5", lw=0.9)
    text(ax, 24, 626, "Configurations (BF16, random weights, embedding dim 128, C = 256 candidates per user)", 9.4,
         weight="bold")
    for i, s_ in enumerate((
            "S:  n = 3 layers,  h = 512,   H × d = 4 × 128  —  KV 6 KiB per token (all layers),  8 MFLOP dense per token",
            "M:  n = 8 layers,  h = 1024, H × d = 4 × 256  —  KV 32 KiB per token,  84 MFLOP dense per token  (main)",
            "L:  n = 8 layers,  h = 2048, H × d = 8 × 256  —  KV 64 KiB per token,  336 MFLOP dense per token")):
        text(ax, 24, 647 + 18 * i, s_, M)
    text(ax, 24, 702, "Black: both strategies.  Orange: remote fetch only (history K/V over RDMA).  Dashed: residual.",
         7.6, SUB)
    text(ax, 24, 716, "Simplified from recsys-examples' HSTU: no contextual features, item and action embeddings summed "
         "(not interleaved tokens), no time encoding.", 7.6, SUB)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "hstu-model.png"))
    print("wrote", os.path.join(out, "hstu-model.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/figs")
