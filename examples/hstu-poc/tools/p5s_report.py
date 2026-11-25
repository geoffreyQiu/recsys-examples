#!/usr/bin/env python3
"""P5s report: serving throughput within the SLO, each strategy alone (overlap-optimized servers).
    tools/p5s_report.py results/p5s -o docs/figs/p5s-capacity.png --curves docs/figs/p5s-curves.png [--md]
Capacity = the highest offered load (users/s, 1 GPU) whose p99 latency is within the SLO with >= 99% of requests
within it and >= 90% of the offered load achieved (a 5 s Poisson sample varies by a few %).
Rows tagged p5s-fine (tools/p5s_fine.sh: 6% load steps, ~1500 requests per point) supersede the coarse rows of the same
(config, strategy, history length). Figures and the main table compare recompute with CPU verbs (the RDMA baseline);
the coarse grid's table lists every mechanism."""
import argparse
import glob
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STYLE = {  # fixed categorical slots: recompute = orange; RDMA: CPU verbs (baseline) blue, raw GDAKI aqua, GIN-GDAKI violet
    "recompute": ("#eb6834", "s", "recompute"),
    "verbs": ("#2a78d6", "o", "fetch: CPU verbs (RDMA baseline)"),
    "raw": ("#1baf7a", "^", "fetch: raw GDAKI"),
    "gdaki": ("#6250d6", "D", "fetch: GIN-GDAKI"),
    "proxy": ("#eda100", "v", "fetch: GIN-Proxy"),
}
ORDER = ["recompute", "verbs", "raw", "gdaki", "proxy"]
MAIN = ("recompute", "verbs")


def load(d):
    rows = []
    for f in glob.glob(f"{d}/*.jsonl"):
        if "smoke" in f:
            continue
        rows += [json.loads(x) for x in open(f) if x.strip()]
    fine = {(r["config"], r["strategy"], r["L"]) for r in rows if r.get("tag") == "p5s-fine"}
    main = [r for r in rows if r.get("tag") == "p5s-fine" or (r["config"], r["strategy"], r["L"]) not in fine]
    coarse = [r for r in rows if r.get("tag") != "p5s-fine"]
    return main, coarse


def capacity(points, slo):
    ok = [r for r in points if r["p99_ms"] <= slo and r["success"] >= 0.99 and r["achieved"] >= 0.9 * r["offered"]]
    return max(ok, key=lambda r: r["offered"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--curves", default=None)
    ap.add_argument("--curve-lengths", default="4096,16384")
    ap.add_argument("--slo", type=float, default=100.0)
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--queue", default=None, help="figure: capacity share vs time per request, against an ideal "
                    "first-come-first-served queue (M/D/1, simulated)")
    a = ap.parse_args()
    rows, coarse = load(a.dir)
    # recompute: keep the better scheduling policy per (config, L): dynamic batching or one request at a time
    rc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["strategy"] in ("recompute", "recompute_b1"):
            rc[(r["config"], r["L"])][r["strategy"]].append(r)
    best_pol = {}
    for k, d in rc.items():
        caps = {p: (capacity(v, a.slo) or {"offered": 0})["offered"] for p, v in d.items()}
        best_pol[k] = max(caps, key=caps.get)
    rows = [r for r in rows if r["strategy"] not in ("recompute", "recompute_b1")
            or r["strategy"] == best_pol[(r["config"], r["L"])]]
    for r in rows:
        if r["strategy"] == "recompute_b1":
            r["strategy"] = "recompute"
    by = defaultdict(list)
    for r in rows:
        by[(r["config"], r["strategy"], r["L"])].append(r)
    cfgs = [c for c in ("M", "L") if any(k[0] == c for k in by)]
    strats = [s for s in ORDER if any(k[1] == s for k in by) and s in MAIN]
    cap = {k: capacity(v, a.slo) for k, v in by.items()}
    fig, axes = plt.subplots(1, len(cfgs), figsize=(5.6 * len(cfgs), 3.9), dpi=140, facecolor=SURFACE, squeeze=False)
    for ax, c in zip(axes[0], cfgs):
        ax.set_facecolor(SURFACE)
        Ls = sorted({k[2] for k in by if k[0] == c})
        for s in strats:
            pts = [(L, cap[(c, s, L)]["offered"]) for L in Ls if cap.get((c, s, L))]
            if not pts:
                continue
            col, mk, name = STYLE[s]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=5, lw=1.8, mec=SURFACE, mew=1,
                    label=name, zorder=3)
        # ceilings: NIC (fetch) and GPU (recompute), from the servers' start-up measurements
        nic = sorted({(r["L"], r["nic_ceiling"]) for r in rows if r["config"] == c and r["kind"] == "fetch"})
        gpu = sorted({(r["L"], r["gpu_ceiling"]) for r in rows if r["config"] == c and r["kind"] == "recompute"})
        if nic:
            ax.plot([p[0] for p in nic], [p[1] for p in nic], color=STYLE["verbs"][0], lw=1, ls=(0, (4, 3)), zorder=2,
                    label="NIC ceiling (49.4 GB/s ÷ KV bytes per user)")
        if gpu:
            ax.plot([p[0] for p in gpu], [p[1] for p in gpu], color=STYLE["recompute"][0], lw=1, ls=(0, (4, 3)),
                    zorder=2, label="recompute ceiling (best batch)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(Ls)
        ax.set_xticklabels([f"{L / 1024:g}k" for L in Ls], fontsize=8, color=INK2)
        ax.set_xlabel("history length (tokens)", color=INK2, fontsize=8.5)
        ax.set_title(f"config {c}", color=INK, fontsize=10, loc="left")
        ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel(f"users/s within the SLO (p99 ≤ {a.slo:.0f} ms, 1 GPU)", color=INK2, fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.12))
    fig.suptitle("Serving capacity within the SLO, each strategy alone (open-loop Poisson load, 256 candidates)",
                 color=INK, fontsize=10, y=1.17, x=0.01, ha="left")
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", a.out)
    if a.curves:
        CL = [int(x) for x in a.curve_lengths.split(",")]
        fig, axes = plt.subplots(len(cfgs), len(CL), figsize=(4.4 * len(CL), 3.0 * len(cfgs)), dpi=140,
                                 facecolor=SURFACE, squeeze=False)
        capy = 3 * a.slo
        for i, c in enumerate(cfgs):
            for j, L in enumerate(CL):
                ax = axes[i][j]
                ax.set_facecolor(SURFACE)
                for s in strats:
                    pts = sorted(by.get((c, s, L), []), key=lambda r: r["offered"])
                    if not pts:
                        continue
                    col, mk, name = STYLE[s]
                    ax.plot([r["offered"] for r in pts], [min(r["p99_ms"], capy) for r in pts], color=col, marker=mk,
                            ms=4.5, lw=1.6, mec=SURFACE, mew=1, label=name, zorder=3)
                ax.axhline(a.slo, color=INK2, lw=1, ls=(0, (4, 3)))
                ax.set_ylim(0, capy * 1.04)
                ax.set_xlim(left=0)
                ax.set_title(f"config {c}, history {L // 1024}k", color=INK, fontsize=9.5, loc="left")
                ax.grid(True, color=GRID, lw=0.6)
                ax.tick_params(colors=INK2, labelsize=7.5)
                for sp in ax.spines.values():
                    sp.set_color(GRID)
                if i == len(cfgs) - 1:
                    ax.set_xlabel("offered load (users/s, 1 GPU)", color=INK2, fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"p99 latency (ms, capped at {capy:.0f})", color=INK2, fontsize=8.5)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", ncol=4, frameon=False, fontsize=8, labelcolor=INK, bbox_to_anchor=(0.01, 1.06))
        fig.tight_layout()
        fig.savefig(a.curves, bbox_inches="tight", facecolor=SURFACE)
        print("wrote", a.curves)
    if a.queue:
        # M/D/1 bound: Poisson arrivals, a fixed service time D per request, served in arrival order. Highest load
        # (share of the ceiling 1/D) whose p99 response time (wait + D) is within the SLO; Lindley recursion.
        import numpy as np
        E = np.random.default_rng(1).exponential(1.0, 400_000)

        def p99_resp(rho, D):
            S = np.cumsum(D - E * (D / rho))
            return float(np.percentile(S - np.minimum.accumulate(np.minimum(S, 0.0)), 99)) + D

        Ds = np.logspace(np.log10(0.3), np.log10(60), 48)
        bound = []
        for D in Ds:
            lo, hi = 0.0, 0.999
            for _ in range(16):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if p99_resp(mid, D) <= a.slo else (lo, mid)
            bound.append(lo)
        fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=140, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(Ds, bound, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=2,
                label=f"ideal first-come-first-served queue (M/D/1), p99 ≤ {a.slo:.0f} ms")
        for s in strats:
            col, mk, name = STYLE[s]
            for c, filled in (("M", True), ("L", False)):
                pts = [(1e3 / r["ceiling"], r["offered"] / r["ceiling"]) for (cc, ss, _), r in cap.items()
                       if cc == c and ss == s and r]
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=34, lw=1.3, zorder=3,
                               color=col if filled else SURFACE, edgecolor=col, label=f"{name}, config {c}")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("time one request occupies the bottleneck: NIC (fetch) or GPU (recompute), ms", color=INK2,
                      fontsize=8.5)
        ax.set_ylabel("capacity within the SLO ÷ ceiling", color=INK2, fontsize=8.5)
        ax.grid(True, which="both", color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK)
        ax.set_title("Share of the ceiling usable within the SLO, against the length of one request", color=INK,
                     fontsize=9.5, loc="left")
        fig.savefig(a.queue, bbox_inches="tight", facecolor=SURFACE)
        print("wrote", a.queue, "; M/D/1 bound at D = 8 / 11 / 16 / 22 / 36 ms:",
              " / ".join(f"{np.interp(np.log(d), np.log(Ds), bound):.2f}" for d in (8, 11, 16.3, 21.7, 35.7)))
    if a.md:
        for c in cfgs:
            Ls = sorted({k[2] for k in by if k[0] == c})
            print(f"\n**Config {c}: users/s within the SLO (share of the strategy's ceiling; GPU busy share; transport cores)**\n")
            print("| Strategy | " + " | ".join(f"{L / 1024:g}k" for L in Ls) + " |")
            print("|---|" + "---|" * len(Ls))
            for s in strats:
                cells = []
                for L in Ls:
                    r = cap.get((c, s, L))
                    if r:
                        cells.append(f"**{r['offered']:,.0f}** ({100 * r['offered'] / r['ceiling']:.0f}%; "
                                     f"GPU {100 * r['gpu_share']:.0f}%; {r['cores_other']:.2f} c)")
                    else:
                        cells.append("–")
                print(f"| {STYLE[s][2]} | " + " | ".join(cells) + " |")
            # SLO break-even: first length where the best fetch capacity >= recompute capacity (log interpolation)
            ratio = []
            for L in Ls:
                rc = cap.get((c, "recompute", L))
                fb = max([cap[(c, s, L)]["offered"] for s in strats if s != "recompute" and cap.get((c, s, L))] or [0])
                ratio.append((L, fb / rc["offered"] if rc and fb else None))
            be = None
            pts = [(L, r) for L, r in ratio if r]
            for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
                if r0 < 1 <= r1:
                    be = math.exp(math.log(l0) + (0 - math.log(r0)) * (math.log(l1) - math.log(l0)) / (math.log(r1) - math.log(r0)))
            print(f"\nbest fetch / recompute capacity: " + ", ".join(f"{L / 1024:g}k {r:.2f}" for L, r in pts)
                  + (f"; SLO break-even ≈ {be / 1024:.1f}k" if be else ""))
        # every mechanism on the coarse grid (fetch only; the fine grid ran CPU verbs)
        cby = defaultdict(list)
        for r in coarse:
            if r["kind"] == "fetch":
                cby[(r["config"], r["strategy"], r["L"])].append(r)
        ccap = {k: capacity(v, a.slo) for k, v in cby.items()}
        print("\n**Every RDMA mechanism, coarse load grid: users/s within the SLO**\n")
        cLs = sorted({k[2] for k in cby})
        print("| Config | Mechanism | " + " | ".join(f"{L / 1024:g}k" for L in cLs) + " |")
        print("|---|---|" + "---|" * len(cLs))
        for c in cfgs:
            for s in ORDER[1:]:
                if not any(k[0] == c and k[1] == s for k in cby):
                    continue
                print(f"| {c} | {STYLE[s][2].replace('fetch: ', '')} | " + " | ".join(
                    f"{ccap[(c, s, L)]['offered']:,.0f}" if ccap.get((c, s, L)) else "–" for L in cLs) + " |")
        print("\nrecompute policy used (higher capacity): " + ", ".join(
            f"{c} {L / 1024:g}k {'one at a time' if p == 'recompute_b1' else 'batched'}" for (c, L), p in sorted(best_pol.items())))
        n_chk = sum(r.get("checks", 0) for r in rows)
        n_bad = sum(r.get("bad_words", 0) for r in rows)
        st = [r["selftest"] for r in rows if r.get("selftest") is not None]
        print(f"\nData checks: {n_chk} sampled batches, {n_bad} mismatched words; ring self-tests: {len(set(map(str, st)))} "
              f"distinct results {sorted(set(st))}")


if __name__ == "__main__":
    main()
