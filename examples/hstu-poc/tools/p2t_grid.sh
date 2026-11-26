#!/bin/bash
# P2t: best plain-run latency of every RDMA mechanism x wait variant x latency option, against recompute and the
# basic implementation. Host clock (request issued -> output seen), isolated requests, one CUDA graph per cell.
#   tools/p2t_grid.sh JOB small|full "verbs.b.5a raw.t.fused ..."     (one process per mode on GPU 0 / NIC 0)
#   tools/p2t_grid.sh JOB small|full recompute                        (recompute + local on GPU 7 of node A)
# mode = transport.level.variant: transport verbs|raw|gdaki|proxy; level b (basic), b2 (basic again: run-to-run
# noise), t (--tuned: every option), zc (GDAKI zero-copy descriptor only), spin (verbs spinning worker only),
# ac (GIN adaptive contexts only); variant 5a|pdl|pdl_early|fused|tile (P4 wait variants; tile = per-tile waits on
# every mechanism). Builds: ext_v18 / attn_v6 (per-tile v2 and raw GDAKI v3, levels t2 and pl: docs/sol.md); ext_v13 / attn_v2 (raw GDAKI
# per-tile); ext_v12 (GIN per-tile); earlier P2t runs used ext_v8 / attn and ext_v9 / attn_v2 (CPU verbs per-tile).
set -u
J=$1; GRIDN=$2; MODES=$3
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=${P2T_OUT:-$D/hstu-kv-rdma/results/p2t}   # P2T_OUT: e.g. results/p2t_ctl for same-allocation controls
S=$POC/scripts/srun2.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v18}
AB=${HPOC_ATTN_BUILD_DIR:-$POC/build/attn_v6}
case $GRIDN in
  small) CELLS=("--configs S,M,L --L 512,1024,1536,2048,3072,4096 --B 1,2,4,8 --budget-s 1.0 --min-iters 50 --max-iters 1000") ;;
  full)  CELLS=("--configs S,M,L --L 6144,8192,12288,16384,24576,32768 --B 1,2,4,8,16,32 --budget-s 1.0 --min-iters 10 --max-iters 300"
               "--configs S,M,L --L 512,1024,1536,2048,3072,4096 --B 16,32 --budget-s 1.0 --min-iters 10 --max-iters 300") ;;
esac
COMMON="--C 256 --isolated --clock host"
POOL="--pool-gb 40 --local-gb 40"
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 5400 "$@" 2>&1 | grep -v -E "$F" | grep -E '^P2|rror|SKIP|check [1-9]' ; }
for m in $MODES; do
  for c in "${CELLS[@]}"; do
    if [ "$m" = recompute ]; then
      run $S -j $J -1 -e HPOC_LOCAL_OFFSET=7 -e OMP_NUM_THREADS=16 -e HPOC_ATTN_BUILD_DIR=$AB -- python bench/grid.py --modes recompute,local \
          $c $COMMON --mem-gb 76 --out $R/${GRIDN}_recompute --tag p2t-$GRIDN
      continue
    fi
    IFS=. read -r T LV V <<< "$m"
    E=(); TR="--transport $T --nqp 4"
    case $T in gdaki) E=(-e NCCL_GIN_TYPE=3); TR="--transport gin" ;; proxy) E=(-e NCCL_GIN_TYPE=2); TR="--transport gin" ;; esac
    X=()   # t2: per-tile v2 (CPU verbs split posting + relay v2; raw GDAKI post kernel v2), see tools/tl_grid.sh
    case $LV in t) O="--tuned" ;; zc) O="--zc" ;; spin) O="--spin" ;; ac) O="--adapt-ctx" ;;
      t2) O="--tuned"; X=(-e HPOC_SPLIT_POST=1 -e HPOC_RELAY=2 -e HPOC_RAW_POST=2) ;;
      pl) O="--tuned"; X=(-e HPOC_RAW_POST=2 -e HPOC_RAW_PRELAUNCH=1) ;; *) O="" ;; esac   # pl: raw GDAKI v3
    run $S -j $J -n 1 -e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16 -e HPOC_ATTN_BUILD_DIR=$AB "${E[@]}" "${X[@]}" -- python bench/p2.py $TR --label $m \
        --modes $m --p4 $V $O $c $COMMON $POOL --out $R/${GRIDN}_$m --tag p2t-$GRIDN
  done
done
echo "== p2t $GRIDN done: $MODES $(date +%T)"
