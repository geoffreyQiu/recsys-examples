#!/bin/bash
# P5s: throughput within the SLO, each strategy alone, overlap-optimized servers (hstu_poc/serving2.py).
# One process per (strategy, config, history length). Fetch runs use GPU 0 / NIC 0 of node A and the store on node B;
# recompute runs use GPU 7 of node A (independent GPUs), so a fetch part and a recompute part can run at once.
#   tools/p5s_runs.sh JOB fetch "M" "verbs raw gdaki" ["1024 2048 ..."]
#   tools/p5s_runs.sh JOB recompute "M L" ["1024 ..."]
set -u
J=$1; PART=$2; CFGS=$3
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p5s
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
FR=${FRACS:-0.4,0.55,0.65,0.72,0.78,0.84,0.89,0.93,0.96,1.0}
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 2400 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP|rc=|GPU core'; }
if [ "$PART" = fetch ]; then
  TRS=$4; LS=${5:-"1024 2048 4096 8192 12288 16384"}
  for c in $CFGS; do for t in $TRS; do for L in $LS; do
    case $t in
      gdaki) E=(-e NCCL_GIN_TYPE=3); TA="--transport gin --label gdaki" ;;
      proxy) E=(-e NCCL_GIN_TYPE=2); TA="--transport gin --label proxy" ;;
      *) E=(); TA="--transport $t --label $t" ;;
    esac
    run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 "${E[@]}" -- $W python bench/p5s.py --strategy fetch $TA \
        --config $c --L $L --fracs $FR --out $R/fetch_$c --tag p5s
  done; done; done
else
  LS=${4:-"1024 2048 4096 8192 12288 16384"}
  for c in $CFGS; do for L in $LS; do
    run $S -j $J -1 -e HPOC_LOCAL_OFFSET=7 -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy recompute \
        --config $c --L $L --fracs $FR --out $R/recompute_$c --tag p5s
  done; done
fi
echo "== p5s $PART $CFGS done $(date +%T)"
