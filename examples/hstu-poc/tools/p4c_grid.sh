#!/bin/bash
# P4c: kernel-boundary variants on small and medium requests, including pdl_early (early CTA release).
# First the slow-producer correctness check per transport, then the latency grid.   tools/p4c_grid.sh JOB
set -u
J=$1; TRS=${2:-"raw gdaki verbs"}
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p4c
S=$POC/scripts/srun2.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=$POC/build/ext_v5
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
CELLS="--configs S,M,L --L 512,1024,2048,4096 --B 1,4 --C 256 --isolated --budget-s 1.5 --min-iters 50 --max-iters 400 --pool-gb 40 --local-gb 40"
for T in $TRS; do
  case $T in
    gdaki) TR="--transport gin --label gdaki"; E=(-e NCCL_GIN_TYPE=3); TC="gin" ;;
    raw)   TR="--transport raw --nqp 4 --label raw"; E=(); TC="raw" ;;
    verbs) TR="--transport verbs --nqp 4 --label verbs"; E=(); TC="verbs" ;;
  esac
  echo "== check $T $(date +%T)"
  timeout 900 $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=8 "${E[@]}" -- python bench/p4_check.py --transport $TC \
    2>&1 | grep -v -E "$F" | grep -E 'PASS|FAIL|slow-producer|rror'
  for V in 5a pdl fused pdl_early; do
    echo "== $T $V $(date +%T)"
    timeout 2400 $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=8 "${E[@]}" -- python bench/p2.py $TR --p4 $V \
      --modes $T $CELLS --out $R/${T}_$V --tag p4c-$V 2>&1 | grep -v -E "$F" | grep -E '^\[r0\]|rror' | tail -3
  done
done
echo "== p4c done $(date +%T)"
