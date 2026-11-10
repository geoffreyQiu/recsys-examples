#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run.sh preflight|no_cache|gpu_hit|cpu_hit|ssd_hit|ssd_layerwise [benchmark args...]
mode=${1:-preflight}
shift || true
repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${GPUS:?Set GPUS=device=2 or device=3 for an available RTX PRO 5500}"
image=${IMAGE:-hstu-flexkv-minimal:sm120}
results=${RESULTS_DIR:-$PWD/results-flexkv}
mkdir -p "$results"
results=$(realpath "$results")
record_dir=${PERF_RECORD_DIR:-$repo/perf}
mkdir -p "$results/runs"
results_root=$results
results=$(mktemp -d "$results/runs/${mode}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "$results" > "$results_root/latest-run.txt"
echo "[perf] Live log: $results/$mode.log"
run_id=$(basename "$results")
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free --format=csv > "$results/gpus.csv" 2>&1 || true
docker image inspect --format '{{.Id}}' "$image" > "$results/image-id.txt" 2>&1 || true
git -C "$repo" rev-parse HEAD > "$results/source-revision.txt" 2>&1 || true
sha256sum "$repo/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py" > "$results/benchmark.sha256"
finish_run() {
    local code=$1
    local profile_args=()
    if [[ ${PROFILE:-0} == 1 ]]; then profile_args+=(--profile); fi
    python3 "$repo/docker/flexkv-minimal/collect_perf.py" \
        --log "$results/$mode.log" --mode "$mode" --run-id "$run_id" \
        --run-dir "$results" --record-dir "$record_dir" --exit-code "$code" \
        --image "$image" --gpus "${GPUS:-device=0}" "${profile_args[@]}"
    echo "[perf] Raw artifacts: $results"
}
run=(docker run --rm --gpus "${GPUS:-device=0}" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
     -v "$results:/results"
     -v "$repo/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py:/workspace/recsys-examples/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py:ro")
if [[ $mode == preflight ]]; then
    set +e
    "${run[@]}" "$image" python /workspace/recsys-examples/docker/flexkv-minimal/preflight.py 2>&1 | tee "$results/$mode.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    code=${pipeline_status[0]}
    if [[ $code == 0 && ${pipeline_status[1]} != 0 ]]; then code=${pipeline_status[1]}; fi
    finish_run "$code"
    exit "$code"
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
# Synchronized forward wall samples are printed by the mounted benchmark.
# PROFILE=1 additionally captures the NVTX ranges and CUDA timeline.
if [[ ${PROFILE:-0} == 1 ]]; then
    cmd=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
         --force-overwrite=true -o "/results/$mode" "${cmd[@]}")
fi
printf '%q ' "${run[@]}" "$image" "${cmd[@]}" > "$results/command.txt"
printf '\n' >> "$results/command.txt"
set +e
"${run[@]}" "$image" "${cmd[@]}" 2>&1 | tee "$results/$mode.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
code=${pipeline_status[0]}
if [[ $code == 0 && ${pipeline_status[1]} != 0 ]]; then code=${pipeline_status[1]}; fi
finish_run "$code"
exit "$code"
