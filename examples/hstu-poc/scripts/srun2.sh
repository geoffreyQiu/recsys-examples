#!/bin/bash
# Launch a POC program on a 2-node allocation, N ranks per node: node A (first) = inference ranks [0, N),
# node B = store ranks [N, 2N); rank i pairs with i + N, GPU = local rank.
#   scripts/srun2.sh -j JOBID [-n N] [-e VAR=value]... -- python bench/x.py [args]
#   scripts/srun2.sh -j JOBID -1 -- python ...      (one rank on node A only)
set -u
D=/lustre/fsw/coreai_devtech_all/junyiq
POC=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
IMG=${IMG:-$D/images/pytorch-26.07.sqsh}
J=; N=1; ONE=0; ENVS=()
while [ $# -gt 0 ]; do
  case $1 in
    -j) J=$2; shift 2 ;; -n) N=$2; shift 2 ;; -e) ENVS+=("$2"); shift 2 ;; -1) ONE=1; shift ;;
    --) shift; break ;; *) break ;;
  esac
done
[ -n "$J" ] || { echo "need -j JOBID"; exit 2; }
read -r A B < <(scontrol show hostnames "$(squeue -j "$J" -h -o %N)" | xargs)
PYP=$POC:$D/hstu-kv-rdma/repos/FBGEMM/fbgemm_gpu/experimental/hstu
if [ "$ONE" = 1 ]; then LAYOUT=(-N1 -n1 -w "$A"); else LAYOUT=(-N2 -n$((2 * N)) --ntasks-per-node="$N" -w "$A,$B"); fi
exec srun --jobid="$J" --overlap "${LAYOUT[@]}" --container-image="$IMG" --container-mounts="$D:$D" \
  --container-workdir="$POC" --export=ALL,MASTER_ADDR="$A",MASTER_PORT="${MASTER_PORT:-29511}" \
  "$D/run.sh" env PYTHONPATH="$PYP" HPOC_BUILD_DIR="${HPOC_BUILD_DIR:-$POC/build/ext}" "${ENVS[@]}" "$@"
