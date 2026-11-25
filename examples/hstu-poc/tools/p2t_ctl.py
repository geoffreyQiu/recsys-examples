#!/usr/bin/env python3
"""Same-pair comparisons for the per-tile waits on every mechanism.
    tools/p2t_ctl.py results/p2t results/p2t_ctl/A results/p2t_ctl/B [results/p2t_tile_ext11 results/p2t_tile_v1
                     [results/p2t_post]]
Each GDAKI/GIN per-tile grid in results/p2t was measured on one GPU/NIC pair next to controls on that same pair:
pair A: raw GDAKI and GIN-Proxy per-tile, controls CPU verbs per-tile, raw GDAKI fused, GIN-Proxy fused;
pair B: GIN-GDAKI per-tile, controls CPU verbs per-tile, GIN-GDAKI fused.
Latency = host-clock mean on the small grid (S/M/L, history 0.5k-4k, B 1-8). 'faster' = beyond the run-to-run noise
(0.95%, the 95th percentile of a repeated run).
Optional: the posting ablations of raw GDAKI per-tile: one batch per layer and queue pair vs one DOCA get per chunk
(pair A; both builds with a separate publisher kernel), and warp-parallel vs one-thread batches (results/p2t_post:
warp/, thread/, two interleaved runs each on pair B)."""
import glob
import json
import statistics
import sys

NOISE = 0.0095
NAME = {"raw": "raw GDAKI", "gdaki": "GIN-GDAKI", "proxy": "GIN-Proxy", "verbs": "CPU verbs"}


def load(d):
    """{(mode, config, B, L): host-clock mean}, averaged over repeated runs of a cell."""
    rows = {}
    for f in glob.glob(f"{d}/small_*.jsonl"):
        for x in open(f):
            r = json.loads(x)
            if "skipped" not in r and "host_mean_ms" in r:
                rows.setdefault((r["mode"], r["config"], int(r["B"]), int(r["L"])), []).append(r["host_mean_ms"])
    return {k: statistics.mean(v) for k, v in rows.items()}


def pick(src, m):
    return {k[1:]: v for k, v in src.items() if k[0] == m}


def compare(name, a, b):
    """a, b: {cell: ms}. One table row: how a compares with b (negative = a faster)."""
    d = [(a[c] / b[c] - 1, c) for c in a if c in b]
    if not d:
        return
    fa = sum(1 for x, _ in d if x < -NOISE)
    fb = sum(1 for x, _ in d if x > NOISE)
    lo, hi = min(d), max(d)
    cell = lambda c: f"{c[0]}, B={c[1]}, {c[2] / 1024:g}k"  # noqa: E731
    print(f"| {name} | {100 * statistics.median(x for x, _ in d):+.1f}% | {100 * lo[0]:+.1f}% ({cell(lo[1])}) | "
          f"{100 * hi[0]:+.1f}% ({cell(hi[1])}) | {fa} / {len(d) - fa - fb} / {fb} |")


def main():
    main_lat, ca, cb = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
    pair = {"raw": ca, "proxy": ca, "gdaki": cb}   # the controls measured on each per-tile grid's pair
    print("| Comparison (same GPU/NIC pair) | median | most negative | most positive | "
          "first faster / tie / second faster |")
    print("|---|---|---|---|---|")
    for m in ("raw", "gdaki", "proxy"):
        compare(f"{NAME[m]} per-tile vs CPU verbs per-tile", pick(main_lat, f"{m}.t.tile"), pick(pair[m], "verbs.t.tile"))
    for m in ("raw", "gdaki", "proxy"):
        compare(f"{NAME[m]} per-tile vs {NAME[m]} fused wait", pick(main_lat, f"{m}.t.tile"), pick(pair[m], f"{m}.t.fused"))
    if len(sys.argv) > 5:
        compare("raw GDAKI per-tile, one batch per layer vs one get per chunk", pick(load(sys.argv[4]), "raw.t.tile"),
                pick(load(sys.argv[5]), "raw.t.tile"))
    if len(sys.argv) > 6:
        compare("raw GDAKI per-tile, warp-parallel vs one-thread batches", pick(load(sys.argv[6] + "/warp"), "raw.t.tile"),
                pick(load(sys.argv[6] + "/thread"), "raw.t.tile"))
    compare("CPU verbs per-tile, pair A vs pair B (pair-to-pair difference)", pick(ca, "verbs.t.tile"),
            pick(cb, "verbs.t.tile"))
    compare("CPU verbs per-tile, pair A vs the main grid's run", pick(ca, "verbs.t.tile"),
            pick(main_lat, "verbs.t.tile"))

    # per cell: the fastest GPU-posted per-tile mechanism against CPU verbs per-tile on its own pair
    rel = {m: {c: v / pair[m][("verbs.t.tile",) + c] - 1 for c, v in pick(main_lat, f"{m}.t.tile").items()
               if ("verbs.t.tile",) + c in pair[m]} for m in ("raw", "gdaki", "proxy")}
    cells = sorted(set().union(*rel.values()))
    print("\n**Fastest GPU-posted per-tile mechanism vs CPU verbs per-tile on the same pair** "
          "(r = raw GDAKI, g = GIN-GDAKI, p = GIN-Proxy; '·' = within noise)\n")
    for cf in ("S", "M", "L"):
        Ls = sorted({c[2] for c in cells if c[0] == cf})
        Bs = sorted({c[1] for c in cells if c[0] == cf})
        if not Ls:
            continue
        print(f"\n{cf}:\n\n| B \\ history | " + " | ".join(f"{L / 1024:g}k" for L in Ls) + " |")
        print("|---|" + "---|" * len(Ls))
        for B in Bs:
            row = []
            for L in Ls:
                c = (cf, B, L)
                opts = [(rel[m][c], m) for m in rel if c in rel[m]]
                if not opts:
                    row.append("–")
                    continue
                x, m = min(opts)
                row.append(f"{m[0]} {100 * x:+.1f}%" if abs(x) > NOISE else f"· ({m[0]} {100 * x:+.1f}%)")
            print(f"| {B} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
