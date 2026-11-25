#!/bin/bash
# P2c: break-even grid for workload R (recompute vs every transport), one harness for all: isolated requests, one
# CUDA graph per cell, GPU-event timing, mean/std/p50/p90/p99 per cell. Configs S/M/D x B 1..32 x L 512..32k.
# recompute + local run on all 16 GPUs of the allocation (cells spread round-robin, GPUs independent); each
# transport runs on GPU 0 of node A against the store on node B.     tools/p2c_grid.sh JOB [steps]
set -u
J=$1; STEPS=${2:-"recompute raw gdaki verbs proxy"}
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p2c
S=$POC/scripts/srun2.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=$POC/build/ext_v5
CFGS=${CFGS:-M,S,D}
GRID="--configs $CFGS --L 512,1024,1536,2048,3072,4096,6144,8192,12288,16384,24576,32768 --B 1,2,4,8,16,32 --C 256 --isolated --budget-s 1.5 --min-iters 10 --max-iters 300"
POOL="--pool-gb 40 --local-gb 40"
G=(-e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16)
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 5400 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|^P2|rror|SKIP'; }
for st in $STEPS; do
  case $st in
    recompute) run $S -j $J -n 8 -- python bench/grid.py --modes recompute,local $GRID --mem-gb 76 --out $R/recompute --tag p2c ;;
    raw)   run $S -j $J -n 1 "${G[@]}" -- python bench/p2.py --transport raw --nqp 4 --label raw --modes raw $GRID $POOL --out $R/raw --tag p2c ;;
    gdaki) run $S -j $J -n 1 "${G[@]}" -e NCCL_GIN_TYPE=3 -- python bench/p2.py --transport gin --label gdaki --modes gdaki $GRID $POOL --out $R/gdaki --tag p2c ;;
    proxy) run $S -j $J -n 1 "${G[@]}" -e NCCL_GIN_TYPE=2 -- python bench/p2.py --transport gin --label proxy --modes proxy $GRID $POOL --out $R/proxy --tag p2c ;;
    verbs) run $S -j $J -n 1 "${G[@]}" -- python bench/p2.py --transport verbs --nqp 4 --modes verbs $GRID $POOL --out $R/verbs --tag p2c ;;
  esac
done
echo "== p2c done $(date +%T)"
