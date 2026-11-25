#!/bin/bash
# P6s follow-up: the fetch server next to the co-located recompute job, with the server's compute graphs and posts on
# high-priority streams (no partitioning; the co-tenant stays at normal priority).   tools/p6s_prio.sh JOB
set -u
J=$1
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p6s
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 1800 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP|rc=|GPU core'; }
for L in 4096 16384; do for t in raw verbs; do
  run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy fetch \
      --transport $t --label $t --config M --L $L --fracs 0.5,0.75,0.9 --cotenant 4096x8 --priority --out $R/p6s --tag p6s
done; done
echo "== p6s prio done $(date +%T)"
