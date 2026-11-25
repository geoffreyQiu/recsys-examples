#!/bin/bash
# P5b: serving curves with the reworked loop (SLO 100 ms, success rate, sampled data checks). One process per
# (transport, history length), so no process builds a second Server (the old loop crashed there, see results.md P5).
#   tools/p5_runs.sh JOB one|eight [transports] [lengths]
#   one:   1 GPU. raw GDAKI fetch + recompute at every length; the other transports' fetch at 1k/4k/16k
#   eight: 8 GPU pairs at once (node scale), fetch only
set -u
J=$1; MODE=$2; TRS=${3:-"raw gdaki verbs proxy"}; LS=${4:-}
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$D/hstu-kv-rdma/repos/recsys-examples/examples/hstu-poc
R=$D/hstu-kv-rdma/results/p5b
S=$POC/scripts/srun2.sh
W=$D/hstu-kv-rdma/tools/gpu_core_wrap3.sh
mkdir -p $R
cd "$POC"
export HPOC_BUILD_DIR=${HPOC_BUILD_DIR:-$POC/build/ext_v5}
G=(-e NCCL_RMA_DISABLE=1 -e OMP_NUM_THREADS=16)
F='NCCL INFO|UserWarning|warn|UMEM|umem|DMABUF'
run() { echo "== $* $(date +%T)"; timeout 3600 "$@" 2>&1 | grep -v -E "$F" | grep -E '^\[r|^==|rror|SKIP|ault|core'; }
tr_args() {
  case $1 in
    raw)   echo "--transport raw --label raw" ;;
    gdaki) echo "--transport gin --label gdaki" ;;
    proxy) echo "--transport gin --label proxy" ;;
    verbs) echo "--transport verbs --label verbs" ;;
  esac
}
tr_env() { case $1 in gdaki) echo "-e NCCL_GIN_TYPE=3" ;; proxy) echo "-e NCCL_GIN_TYPE=2" ;; *) echo "" ;; esac; }
if [ "$MODE" = one ]; then
  for t in $TRS; do
    if [ "$t" = raw ]; then lens=${LS:-"1024 2048 4096 8192 16384"}; else lens=${LS:-"1024 4096 16384"}; fi
    for L in $lens; do
      run $S -j $J -n 1 "${G[@]}" $(tr_env $t) -- $W python bench/p5.py $(tr_args $t) --strategies fetch --L $L --out $R/one_$t --tag p5b
    done
    if [ "$t" = raw ]; then
      run $S -j $J -n 1 "${G[@]}" -- $W python bench/p5.py $(tr_args raw) --strategies recompute --L ${LS:-1024,2048,4096,8192,16384} --out $R/one_recompute --tag p5b
    fi
  done
else
  lens=${LS:-"4096 16384"}
  for t in $TRS; do
    for L in $lens; do
      run $S -j $J -n 8 "${G[@]}" $(tr_env $t) -- $W python bench/p5.py $(tr_args $t) --strategies fetch --L $L --pool-gb 40 --local-gb 40 --out $R/eight_$t --tag p5b-8gpu
    done
  done
fi
echo "== p5 $MODE done $(date +%T)"
