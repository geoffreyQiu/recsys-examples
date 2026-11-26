#!/bin/bash
# Regenerate every figure of docs/results.md from the result rows (needs matplotlib: run inside the PyTorch container,
# e.g. srun --jobid=J --overlap -N1 -n1 --container-image=... --container-workdir=<task dir> tools/make_figs.sh).
set -eu
cd "$(dirname "$0")/.."
F=docs/figs
# explanatory diagrams (schematic): wait variants over time, who posts and reports progress; one request's per-tile
# fetch as sequence diagrams (CPU verbs v2, raw GDAKI v3; line numbers looked up in the POC source); the model
python tools/fetch_diagrams.py $F
python tools/per_tile_diagrams.py $F
python tools/model_diagram.py $F
# speed of light (docs/sol.md): request timelines, latency / SoL, time split against SoL
python tools/sol_figs.py $F
# plain-run latency (P2t, host clock): per transport its fastest implementation, recompute, local hit
python tools/p2t_to_p2c.py results/p2t work/p2t_flat
python tools/report_figs.py work/p2t_flat --cost $F/cost-per-user.png
python tools/p2c_breakeven.py work/p2t_flat -o $F/breakeven.png --configs S,M,L
python tools/p2t_report.py results/p2t --best $F/best-impl.png --ablation $F/opt-ablation.png --bound $F/rdma-vs-bound.png
# serving: strict SLO (P5s), relaxed SLOs and saturation (P5x), shared GPU (P6s)
python tools/p5s_report.py results/p5s -o $F/capacity.png --curves $F/latency-curves.png --queue $F/queueing.png
python tools/p5x_report.py results/p5s results/p5x -o $F/slo-breakeven.png
python tools/p6s_report.py results/p6s -o $F/contention.png
# writes (P3b)
python tools/p3_report.py results/p3b -o $F/writes.png
