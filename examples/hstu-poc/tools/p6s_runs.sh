#!/bin/bash
# P6s: the overlap-optimized fetch server next to a busy GPU. A co-located recompute job (config M, 4k history,
# batches of 8, back to back on its own stream) shares the GPU; with --green 8 it is confined to the other 124 SMs and
# GDAKI posts run on an 8-SM partition. Offered loads are fractions of the server's own (unshared) ceiling.
#   tools/p6s_runs.sh JOB ["raw verbs"] ["4096 16384"]
set -u
J=$1; TRS=${2:-"raw verbs"}; LS=${3:-"4096 16384"}
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p6s
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 1800 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP|rc=|GPU core'; }
for L in $LS; do for t in $TRS; do for g in none 0 8; do
  if [ "$g" = none ]; then CT=""; else CT="--cotenant 4096x8 --green $g"; fi     # none: no co-tenant (baseline)
  run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy fetch \
      --transport $t --label $t --config M --L $L --fracs 0.5,0.75,0.9 $CT --out $R/p6s --tag p6s
done; done; done
echo "== p6s done $(date +%T)"
