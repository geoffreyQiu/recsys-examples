#!/bin/bash
# P6s: redo the 4k green-context runs with the co-located job's graph captured on its green stream.
set -u
J=$1
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
for t in raw verbs; do
  echo "== $t 4096 green 8 $(date +%T)"
  timeout 1800 $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy fetch \
      --transport $t --label $t --config M --L 4096 --fracs 0.5,0.75,0.9 --cotenant 4096x8 --green 8 \
      --out $D/hstu-kv-rdma/results/p6s/p6s --tag p6s 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|rc='
done
echo "== p6s green4k done $(date +%T)"
