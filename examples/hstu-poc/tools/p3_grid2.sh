#!/bin/bash
# P3b: workload W on a regular grid (M, C = 256): B 1/2/4/8 x prefix P 0/1k/2k/4k/8k/16k x new tokens S
# 256/1k/2k/4k/8k. Per cell: compute only ("none"), and the same forward with per-layer writes of the new K/V.
#   tools/p3_grid2.sh JOB [transports]
set -u
J=$1; TRS=${2:-"gdaki verbs proxy"}
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p3b
S=$POC/scripts/srun2.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=$POC/build/ext_v5
CELLS="--configs M --B 1,2,4,8 --C 256 --fresh= --P 0,1024,2048,4096,8192,16384 --S 256,1024,2048,4096,8192 --budget-s 1 --min-iters 10 --max-iters 200 --pool-gb 40 --local-gb 40"
G=(-e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16)
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 5400 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP'; }
for t in $TRS; do
  case $t in
    gdaki) run $S -j $J -n 1 "${G[@]}" -e NCCL_GIN_TYPE=3 -- python bench/p3.py --transport gin --label gdaki --modes none,gin $CELLS --out $R/gdaki --tag p3b ;;
    proxy) run $S -j $J -n 1 "${G[@]}" -e NCCL_GIN_TYPE=2 -- python bench/p3.py --transport gin --label proxy --modes gin $CELLS --out $R/proxy --tag p3b ;;
    verbs) run $S -j $J -n 1 "${G[@]}" -- python bench/p3.py --transport verbs --modes verbs $CELLS --out $R/verbs --tag p3b ;;
  esac
done
echo "== p3b done $(date +%T)"
