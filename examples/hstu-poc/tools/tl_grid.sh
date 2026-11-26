#!/bin/bash
# Per-request timelines for the speed-of-light gap analysis: each mode is measured as in P2t (host clock, isolated
# requests), then TL_N more requests run with a traced copy of the graph (bench/grid.py --timeline; build ext_v18 /
# attn_v6; earlier campaigns: see docs/sol.md, appendix). Modes as in tools/p2t_grid.sh (transport.level.variant), plus "recompute" (recompute + local hit, GPU 0).
#   tools/tl_grid.sh JOB "verbs.t.tile raw.t.tile verbs.b.5a ... recompute"
# TL_CELLS overrides the cells (grid.py arguments), TL_OUT the results directory, TL_N the traced requests per cell.
set -u
J=$1; MODES=$2
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=${TL_OUT:-$D/hstu-kv-rdma/results/tl}
S=$POC/scripts/srun2.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v18}
AB=${HPOC_ATTN_BUILD_DIR:-$POC/build/attn_v6}
CELLS=${TL_CELLS:-"--configs S,M,L --L 512,1024,4096,16384 --B 1,8"}
COMMON="--C 256 --isolated --clock host --budget-s 1.0 --min-iters 30 --max-iters 300 --timeline ${TL_N:-20}"
POOL="--pool-gb 40 --local-gb 40"
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 3600 "$@" 2>&1 | grep -v -E "$F" | grep -E '^P2|rror|SKIP|check [1-9]|\] [SML] ' ; }
for m in $MODES; do
  if [ "$m" = recompute ]; then
    run $S -j $J -1 -e OMP_NUM_THREADS=16 -e HPOC_ATTN_BUILD_DIR=$AB -- python bench/grid.py --modes recompute,local \
        $CELLS $COMMON --mem-gb 76 --out $R/recompute --tag tl
    continue
  fi
  IFS=. read -r T LV V <<< "$m"
  E=(); TR="--transport $T --nqp 4"
  case $T in gdaki) E=(-e NCCL_GIN_TYPE=3); TR="--transport gin" ;; proxy) E=(-e NCCL_GIN_TYPE=2); TR="--transport gin" ;; esac
  X=()   # t2: every option of P7 (split posting + relay v2 for CPU verbs, post kernel v2 for raw GDAKI); sp, rl, p2: one
  case $LV in t) O="--tuned" ;; t2) O="--tuned"; X=(-e HPOC_SPLIT_POST=1 -e HPOC_RELAY=2 -e HPOC_RAW_POST=2) ;;
    sp) O="--tuned"; X=(-e HPOC_SPLIT_POST=1) ;; rl) O="--tuned"; X=(-e HPOC_RELAY=2) ;; p2) O="--tuned"; X=(-e HPOC_RAW_POST=2) ;;
    t2r) O="--tuned"; X=(-e HPOC_SPLIT_POST=1 -e HPOC_RELAY=2 -e HPOC_RAW_POST=2 -e HPOC_TILE_RELOAD=1) ;;   # attn_v4: no cache
    t3) O="--tuned"; X=(-e HPOC_SPLIT_POST=1 -e HPOC_DIRECT=1 -e HPOC_RAW_POST=2) ;;   # CPU verbs direct (attn_v5)
    pl) O="--tuned"; X=(-e HPOC_RAW_POST=2 -e HPOC_RAW_PRELAUNCH=1) ;;   # raw GDAKI v3: post kernel before the graph
    *) O="" ;; esac
  run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -e HPOC_ATTN_BUILD_DIR=$AB "${E[@]}" "${X[@]}" -- python bench/p2.py $TR \
      --label $m --modes $m --p4 $V $O $CELLS $COMMON $POOL --out $R/$m --tag tl
done
echo "== tl done: $MODES $(date +%T)"
