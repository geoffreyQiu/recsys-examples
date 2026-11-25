#!/bin/bash
# P5x: serving up to saturation, for capacities at relaxed SLOs (p99 targets 100 ms .. 2 s and the highest sustained
# load). Same servers as P5s (hstu_poc/serving2.py); each sweep runs until two load points are past saturation
# (achieved < 97% of offered or p99 > 2 s). Recompute may batch up to half of a 1 s budget (bigger batches than P5s).
#   tools/p5x_runs.sh JOB fetch CFG "Ls" T LEVEL V      T verbs|raw|gdaki|proxy, LEVEL b|t (tuned), V 5a|fused
#   tools/p5x_runs.sh JOB recompute CFG "Ls" GPU batched|b1
set -u
J=$1; PART=$2; C=$3; LS=$4
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p5x
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
FR=0.4,0.55,0.7,0.8,0.87,0.92,0.95,0.97,0.99,1.01,1.03
X="--min-requests 2500 --max-dur 40 --stop sat --sat-p99-ms 2000 --tag p5x"
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=$POC/build/ext_v8
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 3600 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP|rc=|GPU core'; }
for L in $LS; do
  if [ "$PART" = fetch ]; then
    T=$5; LV=$6; V=$7
    case $T in
      gdaki) E=(-e NCCL_GIN_TYPE=3); TA="--transport gin" ;;
      proxy) E=(-e NCCL_GIN_TYPE=2); TA="--transport gin" ;;
      *) E=(); TA="--transport $T" ;;
    esac
    O=""; [ "$LV" = t ] && O="--tuned"
    run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 "${E[@]}" -- $W python bench/p5s.py --strategy fetch \
        $TA --label $T.$LV.$V --p4 $V $O --config $C --L $L --fracs $FR $X --out $R/fetch_$C
  else
    G=$5; POL=$6
    if [ "$POL" = b1 ]; then PA="--max-batch 1 --label recompute_b1"; O=_b1; else PA=""; O=""; fi
    run $S -j $J -1 -e HPOC_LOCAL_OFFSET=$G -e HPOC_MASTER_PORT=$((53100 + G)) -e OMP_NUM_THREADS=16 -- $W python \
        bench/p5s.py --strategy recompute --config $C --L $L --fracs $FR --slo-ms 1000 $PA $X --out $R/recompute_$C$O
  fi
done
echo "== p5x $PART $C $LS ${5:-} ${6:-} ${7:-} done $(date +%T)"
