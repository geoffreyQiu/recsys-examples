#!/bin/bash
# P5s fine grid for long histories: load steps of 6% of the ceiling and ~1500 requests per point (15-45 s), so the
# capacity within the SLO is resolved to a few % where requests are long (12k+). Rows are tagged p5s-fine and supersede
# the coarse rows of the same (config, strategy, history length) in tools/p5s_report.py.
#   tools/p5s_fine.sh JOB fetch CFG "12288 16384"                 (CPU verbs, GPU 0 / NIC 0 of node A, store on node B)
#   tools/p5s_fine.sh JOB recompute CFG "12288 16384" GPU batched|b1   (one GPU of node A)
set -u
J=$1; PART=$2; C=$3; LS=$4
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p5s
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
FR=0.3,0.36,0.42,0.48,0.54,0.6,0.66,0.72,0.78,0.84,0.9,0.95,1.0
X="--min-requests 1500 --max-dur 45 --tag p5s-fine"
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 3000 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|rror|SKIP|rc=|GPU core'; }
for L in $LS; do
  if [ "$PART" = fetch ]; then
    run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy fetch \
        --transport verbs --label verbs --config $C --L $L --fracs $FR --out $R/fine_fetch_$C $X
  else
    G=$5; POL=$6
    if [ "$POL" = b1 ]; then PA="--max-batch 1 --label recompute_b1"; O=_b1; else PA=""; O=""; fi
    run $S -j $J -1 -e HPOC_LOCAL_OFFSET=$G -e OMP_NUM_THREADS=16 -- $W python bench/p5s.py --strategy recompute \
        --config $C --L $L --fracs $FR --out $R/fine_recompute_$C$O $PA $X
  fi
done
echo "== p5s fine $PART $C $LS ${6:-} done $(date +%T)"
