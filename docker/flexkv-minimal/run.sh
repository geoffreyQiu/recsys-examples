#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run.sh preflight|no_cache|gpu_hit|cpu_hit|ssd_hit|ssd_layerwise [benchmark args...]
mode=${1:-preflight}
shift || true
image=${IMAGE:-hstu-flexkv-minimal:sm120}
results=${RESULTS_DIR:-$PWD/results-flexkv}
mkdir -p "$results"
results=$(realpath "$results")
run=(docker run --rm --gpus "${GPUS:-device=0}" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
     -v "$results:/results")
if [[ $mode == preflight ]]; then
    "${run[@]}" "$image" python /workspace/recsys-examples/docker/flexkv-minimal/preflight.py
    exit
fi
case "$mode" in
    no_cache|gpu_hit|cpu_hit) scenario=$mode; extra=() ;;
    ssd_hit|ssd_layerwise)
        : "${SSD_CACHE_DIR:?Set SSD_CACHE_DIR to a dedicated writable directory on your local SSD}"
        mkdir -p "$SSD_CACHE_DIR"
        run+=(-v "$(realpath "$SSD_CACHE_DIR"):/ssd-cache")
        scenario=ssd_hit
        cat > "$results/ssd.yml" <<'YAML'
cpu_cache_gb: 0.2
ssd_cache_gb: 8.0
ssd_cache_dir: /ssd-cache
enable_gds: false
enable_p2p_cpu: false
enable_p2p_ssd: false
enable_3rd_remote: false
YAML
        extra=(--flexkv-config-path /results/ssd.yml --no-prefetch)
        if [[ $mode == ssd_layerwise ]]; then extra+=(--layerwise --layer-granularity "${LAYER_GRANULARITY:-1}"); fi ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
cmd=(python inference/benchmark/inference_benchmark_flexkv.py --scenarios "$scenario"
     --history-len 1024 --append-history-len 0 --only-onboard
     --warmup-iters 100 --timed-iters 50 "${extra[@]}" "$@")
# Upstream GPU/CPU cases expose NVTX timed ranges rather than printed latency.
# PROFILE=1 collects those ranges without changing the benchmark source.
if [[ ${PROFILE:-0} == 1 ]]; then
    cmd=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
         --force-overwrite=true -o "/results/$mode" "${cmd[@]}")
fi
"${run[@]}" "$image" "${cmd[@]}" 2>&1 | tee "$results/$mode.log"
