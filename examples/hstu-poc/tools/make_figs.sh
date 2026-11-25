#!/bin/bash
# Regenerate every figure of docs/results.md from the result rows (needs matplotlib: run inside the PyTorch container,
# e.g. srun --jobid=J --overlap -N1 -n1 --container-image=... --container-workdir=<task dir> tools/make_figs.sh).
set -eu
cd "$(dirname "$0")/.."
F=docs/figs
python tools/report_figs.py results/p2c --cost $F/cost-per-user.png --xport $F/transport-overhead.png
python tools/p2c_breakeven.py results/p2c -o $F/breakeven.png --configs S,M,L
python tools/p5s_report.py results/p5s -o $F/capacity.png --curves $F/latency-curves.png --queue $F/queueing.png
python tools/p6s_report.py results/p6s -o $F/contention.png
python tools/p3_report.py results/p3b -o $F/writes.png
