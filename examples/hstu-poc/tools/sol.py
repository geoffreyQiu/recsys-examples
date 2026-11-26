#!/usr/bin/env python3
"""Speed-of-light (SoL) model of one HSTU ranking request, recompute and remote fetch, on one H100 SXM + one CX-7 NDR.

    tools/sol.py ops CONFIG B L              per-operation roofline table of both workflows
    tools/sol.py grid [results/p2t]          SoL latency of every grid point vs the measured latency (markdown)
    tools/sol.py serving                     SoL serving ceilings and M/D/1 latency at the SoL service time
    tools/sol.py json OUT [results/p2t]      everything above as JSON (for the figures)

Model (docs/sol.md):
- The least operations of a request, fully fused: per layer one projection GEMM (LayerNorm prologue, bias + SiLU
  epilogue, K/V written straight into the KV buffer), one attention (SiLU scores), one output GEMM (LayerNorm(attn) *
  U prologue, bias + residual epilogue); the embedding before the first layer and the candidates' head after the last.
- Each operation's time is its roofline: the largest of tensor FLOPs / 989 TFLOPS, HBM bytes / 3.35 TB/s, SFU ops /
  3.9 Tops/s and FP32 instructions / 33.5 Tinstr/s (operands read once from HBM; weights are not L2-resident).
- Operations of one request depend on each other in sequence (layer l needs layer l - 1), so a request's compute
  time is the sum of its operations' rooflines; overlap across requests is only possible in serving.
- Remote fetch: the history's K/V stream over the NIC at its payload rate (49.7 GB/s at NDR 400 Gb/s and 4 KiB MTU)
  from t_rdma (doorbell + first read's round trip) after the request is issued; the reads need no GPU. The GPU
  starts after t_in (host -> GPU notification) and its compute chain waits only where a K/V tile it needs has not
  landed (tiles consumed in arrival order). The output is visible to the host t_out after the last operation.
"""
import glob
import json
import math
import os
import sys

# ---------------------------------------------------------------- hardware (theoretical peaks)
HW = dict(
    tensor=989.4e12,        # dense BF16 tensor-core FLOP/s, H100 SXM5 (132 SMs, 4096 FLOP/clk/SM at 1.83 GHz)
    hbm=3.35e12,            # HBM3 bytes/s
    sfu=132 * 16 * 1.83e9,  # MUFU ops/s (16 per clock per SM: exp2, rcp, tanh ...)
    fp32=132 * 128 * 1.98e9,  # FP32 instructions/s (128 lanes per SM; an FMA is one instruction)
    nic=50e9 * 4096 / 4122,  # RDMA read payload bytes/s: 400 Gb/s, 4 KiB MTU, 26 B of IB headers per packet
    pcie=63e9,              # PCIe Gen5 x16 per direction (NIC -> GPU path), not a bottleneck (> NIC)
)
# fixed latencies (s), measured on Eos (results/sol_lat): the hardware path, not our software
LAT = dict(
    t_in=1.2e-6,     # host -> GPU: a write to mapped host memory seen by a resident GPU thread (half the median
                     # host-GPU-host round trip, 2.4-2.6 us, tools/pcie_pingpong.cu)
    t_out=1.2e-6,    # GPU -> host: the same path back
    t_rdma=3.35e-6,  # post + one small RDMA read's round trip (ib_read_lat, 2-64 B, typical): the first byte lands
)

CFG = {  # name: layers, hidden, heads, head_dim (hstu_poc/model.py CONFIGS); emb 128, MLP 512-8, C = 256
    "S": (3, 512, 4, 128), "M": (8, 1024, 4, 256), "L": (8, 2048, 8, 256)}
EMB, M1, M2, E = 128, 512, 8, 2   # embedding dim, MLP widths, bytes per BF16 element


def op(name, flops=0.0, hbm=0.0, sfu=0.0, fp32=0.0, layer=None, part=None):
    t = {"tensor": flops / HW["tensor"], "hbm": hbm / HW["hbm"], "sfu": sfu / HW["sfu"], "fp32": fp32 / HW["fp32"]}
    bound = max(t, key=t.get)
    return dict(name=name, flops=flops, hbm=hbm, sfu=sfu, fp32=fp32, t=t[bound], bound=bound, layer=layer, part=part)


def embed_op(T, h, history):
    """Embedding of T tokens: item (+ action for history tokens) gather, 128 -> h projection, position table, LN."""
    gath = T * EMB * E * (2 if history else 1) + T * h * E        # item/action rows + position rows
    return op("embedding", flops=2 * T * EMB * h, hbm=gath + EMB * h * E + T * h * E, fp32=T * h * 10)


def layer_ops(l, cfg, Tq, Tk_new, pairs_hist, pairs_cand, kv_hist_bytes, B):
    """One layer. Tq: tokens computed (queries); pairs_*: (query, key) pairs of the attention per kind;
    kv_hist_bytes: history K/V read by the attention (fetch: the fetched rows; recompute: 0, its K/V are new)."""
    n, h, H, d = cfg
    hd = H * d
    g1 = op("proj GEMM (LN, bias+SiLU)", flops=2 * Tq * h * 4 * hd,
            hbm=Tq * h * E + h * 4 * hd * E + 4 * hd * E + Tq * 4 * hd * E,
            sfu=Tq * 4 * hd, fp32=Tq * (h * 8 + 4 * hd * 4), layer=l, part="g1")
    a_h = op("attention, history keys", flops=4 * hd * pairs_hist, hbm=kv_hist_bytes,
             sfu=H * pairs_hist, fp32=H * pairs_hist * 4, layer=l, part="ah")
    # candidates' own keys (fetch) or every key (recompute): Q, K, V of the tokens computed now read once, O written
    a_c = op("attention, new keys", flops=4 * hd * pairs_cand, hbm=Tq * hd * E + 2 * Tk_new * hd * E + Tq * hd * E,
             sfu=H * pairs_cand, fp32=H * pairs_cand * 4, layer=l, part="ac")
    g2 = op("out GEMM (LN(attn)*U, bias+res)", flops=2 * Tq * hd * h,
            hbm=2 * Tq * hd * E + hd * h * E + h * E + 2 * Tq * h * E, fp32=Tq * (hd * 10 + h * 2), layer=l, part="g2")
    return [g1, a_c, a_h, g2]


def head_op(BC, h):
    return op("head (LN, MLP 512-8, sigmoid)", flops=2 * BC * (h * M1 + M1 * M2),
              hbm=BC * h * E + (h * M1 + M1 * M2) * E + BC * M2 * E, sfu=BC * M2, fp32=BC * (h * 10 + M1 * 2))


def ops_recompute(c, B, L, C=256):
    cfg = CFG[c]
    n, h, H, d = cfg
    T = B * (L + C)
    pairs = B * (L * (L + 1) / 2 + C * (L + 1))   # history causal; each candidate: all history + itself
    out = [embed_op(T, h, True)]
    for l in range(n):
        out += [o for o in layer_ops(l, cfg, T, T, 0, pairs, 0, B) if o["part"] != "ah"]
    return out + [head_op(B * C, h)]


def ops_fetch(c, B, L, C=256):
    cfg = CFG[c]
    n, h, H, d = cfg
    hd = H * d
    Tq = B * C
    out = [embed_op(Tq, h, False)]
    for l in range(n):
        out += layer_ops(l, cfg, Tq, Tq, B * C * L, B * C, B * L * 2 * hd * E, B)
    return out + [head_op(B * C, h)]


def kv_bytes(c, B, L):
    n, h, H, d = CFG[c]
    return B * L * n * 2 * H * d * E


def sol_recompute(c, B, L, C=256):
    return LAT["t_in"] + sum(o["t"] for o in ops_recompute(c, B, L, C)) + LAT["t_out"]


def sol_fetch(c, B, L, C=256, rows=None, per_layer=False, t_rdma=None):
    """Fetch SoL: the NIC streams layer after layer (each layer's history from its end, `rows` rows per step; None =
    continuous); the compute chain waits for the tiles it consumes. per_layer: a layer's attention waits for the
    whole layer. Returns (latency, dict of parts)."""
    n, h, H, d = CFG[c]
    t_rdma = LAT["t_rdma"] if t_rdma is None else t_rdma
    ops = ops_fetch(c, B, L, C)
    layer_bytes = B * L * 2 * H * d * E
    rows = rows or max(1, L // 256)            # "continuous": 256 steps per layer
    steps = math.ceil(L / rows)
    t = LAT["t_in"] + ops[0]["t"]              # GPU starts, embeds the candidates
    t_first = t_rdma
    for o in ops[1:-1]:
        l = o["layer"]
        if o["part"] != "ah":
            t += o["t"]
            continue
        land0 = t_first + l * layer_bytes / HW["nic"]
        if per_layer:
            t = max(t, land0 + layer_bytes / HW["nic"]) + o["t"]
            continue
        for k in range(steps):
            r = min(rows, L - k * rows)
            land = land0 + min(L, (k + 1) * rows) / L * layer_bytes / HW["nic"]
            t = max(t, land) + o["t"] * r / L
    t += ops[-1]["t"] + LAT["t_out"]
    xfer = t_rdma + n * layer_bytes / HW["nic"]
    comp = LAT["t_in"] + sum(o["t"] for o in ops) + LAT["t_out"]
    return t, dict(transfer=xfer, compute=comp, after_last=t - xfer)


# ---------------------------------------------------------------- serving
def per_user_gpu_s(c, L, strategy, C=256):
    """GPU time per user at SoL in a large batch (weights amortized: B = 1024)."""
    B = 1024
    ops = ops_recompute(c, B, L, C) if strategy == "recompute" else ops_fetch(c, B, L, C)
    return sum(o["t"] for o in ops) / B


def ceilings(c, L, C=256):
    f_nic = HW["nic"] / kv_bytes(c, 1, L)
    f_gpu = 1 / per_user_gpu_s(c, L, "fetch", C)
    r_gpu = 1 / per_user_gpu_s(c, L, "recompute", C)
    return dict(fetch=min(f_nic, f_gpu), fetch_nic=f_nic, fetch_gpu=f_gpu, recompute=r_gpu)


def md1_p99(rho, S, n=400000, seed=1):
    """p99 sojourn time of an M/D/1 queue (service time S, load rho), simulated (Lindley)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.exponential(S / rho, size=n)
    w = np.zeros(n)
    for i in range(1, n):                 # waiting time recursion
        w[i] = max(0.0, w[i - 1] + S - a[i])
    return float(np.percentile(w[n // 10:], 99) + S)


def md1_capacity(S, target, tail=0.0):
    """Highest load rho at which the M/D/1 p99 sojourn + tail stays within target (bisection on rho)."""
    if S + tail > target:
        return 0.0
    lo, hi = 0.0, 0.9999
    for _ in range(18):
        mid = (lo + hi) / 2
        if md1_p99(mid, S, n=120000) + tail <= target:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- measured data (results/p2t)
def load_measured(d):
    """{(config, B, L): {mode: host-clock mean ms}} from the P2t grid (recompute, local, every RDMA mode)."""
    m = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for x in open(f):
            r = json.loads(x)
            if "skipped" in r or "host_mean_ms" not in r or str(r["L"]) == "skew":
                continue
            k = (r["config"], int(r["B"]), int(r["L"]))
            m.setdefault(k, {}).setdefault(r["mode"], []).append(r["host_mean_ms"])
    return {k: {mm: sum(v) / len(v) for mm, v in d_.items()} for k, d_ in m.items()}


def best_fetch(modes):
    f = {k: v for k, v in modes.items() if k not in ("recompute", "local")}
    if not f:
        return None, None
    k = min(f, key=f.get)
    return k, f[k]


def fmt_us(s):
    return f"{s * 1e6:.0f} µs" if s < 1e-3 else f"{s * 1e3:.2f} ms"


def cmd_ops(c, B, L):
    for name, ops in (("recompute", ops_recompute(c, B, L)), ("fetch", ops_fetch(c, B, L))):
        print(f"\n### {name}, {c}, B = {B}, L = {L}\n")
        print("| operation | layers | tensor GFLOP | HBM MB | SFU Gops | time per op | bound |")
        print("|---|---|---|---|---|---|---|")
        groups = {}
        for o in ops:
            groups.setdefault(o["name"], []).append(o)
        for nm, os_ in groups.items():
            o = os_[0]
            print(f"| {nm} | {len(os_) if o['layer'] is not None else '–'} | {o['flops'] / 1e9:.3f} | "
                  f"{o['hbm'] / 1e6:.2f} | {o['sfu'] / 1e9:.3f} | {fmt_us(o['t'])} | {o['bound']} |")
        tot = sum(o["t"] for o in ops)
        print(f"\nsum of operation rooflines: {fmt_us(tot)}; tensor-bound share "
              f"{sum(o['t'] for o in ops if o['bound'] == 'tensor') / tot:.0%}")
    t, parts = sol_fetch(c, B, L)
    print(f"\nfetch SoL {fmt_us(t)} (transfer {fmt_us(parts['transfer'])}, compute {fmt_us(parts['compute'])}); "
          f"recompute SoL {fmt_us(sol_recompute(c, B, L))}")


def grid_rows(d):
    meas = load_measured(d)
    rows = []
    for (c, B, L), modes in sorted(meas.items()):
        if c not in CFG:
            continue
        bm, bf = best_fetch(modes)
        f_sol, parts = sol_fetch(c, B, L)
        f_tile, _ = sol_fetch(c, B, L, rows=128)
        f_layer, _ = sol_fetch(c, B, L, per_layer=True)
        rows.append(dict(config=c, B=B, L=L, sol_fetch=f_sol * 1e3, sol_fetch_tile128=f_tile * 1e3,
                         sol_fetch_layer=f_layer * 1e3, sol_transfer=parts["transfer"] * 1e3,
                         sol_fetch_compute=parts["compute"] * 1e3, sol_recompute=sol_recompute(c, B, L) * 1e3,
                         best_fetch=bf, best_mode=bm, verbs_basic=modes.get("verbs.b.5a"),
                         recompute=modes.get("recompute"), local=modes.get("local")))
    return rows


def cmd_grid(d):
    rows = grid_rows(d)
    print("| config | B | L | fetch SoL | best fetch | ÷ SoL | recompute SoL | recompute | ÷ SoL | local | "
          "fetch compute SoL |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    def v(x, fmt="{:.3f}"):
        return "–" if x is None else fmt.format(x)

    def ratio(a, b):
        return "–" if a is None else f"{a / b:.2f}"
    for r in rows:
        f, rc = r["best_fetch"], r["recompute"]
        print(f"| {r['config']} | {r['B']} | {r['L']} | {r['sol_fetch']:.3f} | {v(f)} | {ratio(f, r['sol_fetch'])} | "
              f"{r['sol_recompute']:.3f} | {v(rc)} | {ratio(rc, r['sol_recompute'])} | {v(r['local'])} | "
              f"{r['sol_fetch_compute']:.3f} |")


def breakeven_sol(c, B, kind="latency"):
    """History length where fetch SoL = recompute SoL (latency at batch B) or the serving ceilings cross."""
    def diff(L):
        if kind == "latency":
            return sol_fetch(c, B, L)[0] - sol_recompute(c, B, L)
        cz = ceilings(c, L)
        return cz["recompute"] - cz["fetch"]
    # the crossing above 1k: below ~0.2k fetch also wins (recompute computes history + candidates), not of interest
    lo = 1024
    if diff(lo) < 0:
        return float("nan")
    hi = lo
    while diff(hi) > 0:
        lo, hi = hi, hi * 1.25
        if hi > (1 << 21):
            return float("inf")
    for _ in range(30):
        mid = (lo + hi) / 2
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cmd_serving():
    print("| config | L | fetch ceiling (NIC) | fetch GPU SoL | recompute SoL ceiling | ratio | fetch p99<=100ms |")
    print("|---|---|---|---|---|---|---|")
    for c in CFG:
        for L in (1024, 2048, 4096, 8192, 16384, 32768):
            cz = ceilings(c, L)
            S = 1 / cz["fetch"]
            tail = sol_fetch(c, 1, L)[0] - kv_bytes(c, 1, L) / HW["nic"]
            cap = md1_capacity(S, 0.1, tail) * cz["fetch"]
            print(f"| {c} | {L} | {cz['fetch_nic']:.0f} | {cz['fetch_gpu']:.0f} | {cz['recompute']:.0f} | "
                  f"{cz['fetch'] / cz['recompute']:.2f} | {cap:.0f} |")
    for c in CFG:
        print(f"{c}: SoL throughput break-even {breakeven_sol(c, 1, 'throughput') / 1024:.1f}k, latency break-even "
              f"B=1 {breakeven_sol(c, 1) / 1024:.1f}k, B=8 {breakeven_sol(c, 8) / 1024:.1f}k")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if cmd == "ops":
        cmd_ops(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "grid":
        cmd_grid(sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "results", "p2t"))
    elif cmd == "serving":
        cmd_serving()
    elif cmd == "json":
        d = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, "results", "p2t")
        out = dict(hw=HW, lat=LAT, grid=grid_rows(d),
                   breakeven={c: dict(throughput=breakeven_sol(c, 1, "throughput"),
                                      latency={B: breakeven_sol(c, B) for B in (1, 2, 4, 8, 16, 32)}) for c in CFG})
        json.dump(out, open(sys.argv[2], "w"), indent=1)
