# Minimal HSTU FlexKV benchmark build

This recipe runs `examples/hstu/inference/benchmark/inference_benchmark_flexkv.py` from upstream revision `d03198c35166e8dae907b6bc453f2fa1112700b0`. It preserves model source and adds explicit benchmark timing; see the collection workflow below. Default deployment target: NVIDIA RTX PRO 5500 Blackwell, compute capability 12.0, on Linux x86_64. Build on the deployment server; transfer source through GitHub rather than copying H100-only binaries.

## Chosen stack and scope

Base: `nvcr.io/nvidia/pytorch:26.07-py3` (Python 3.12, PyTorch 2.13 development build, CUDA 13.3.1), matching the upstream Dockerfile. The successful pull of 26.03 establishes registry access; it does not validate this stack or the GPU driver. Your supplied target snapshot shows driver 620.70.06 / CUDA driver API 13.5, sufficiently recent for this stack. Its RTX PRO 5500s are physical GPUs 2 and 3; both were occupied in the initial snapshot, and the user subsequently confirmed both are available. Check availability before each run. GPU 0 is an RTX 6000D, so selecting GPU 0 would benchmark the wrong model.

Before building, run:

```bash
nvidia-smi
export GPUS=device=2  # choose an available RTX PRO 5500, physical index 2 or 3
docker run --rm --gpus "$GPUS" nvcr.io/nvidia/pytorch:26.07-py3 \
  python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); x=torch.randn(128,128,device="cuda",dtype=torch.bfloat16); print((x@x).sum().item())'
```

The target must report capability `(12, 0)` and complete the CUDA operation. Use an NVIDIA driver compatible with this CUDA release and a working NVIDIA Container Toolkit. A pull alone does not check either. If the preflight fails, resolve the target driver/runtime before spending time compiling. Do not substitute an older base while keeping the same binary FBGEMM wheel: its PyTorch ABI is version coupled.

Included:

- Upstream-matched FBGEMM GPU 1.8.0 CUDA 13.0 wheel (checksum verified) and TorchRec v1.8.0-rc1.
- Dynamic embedding Python/CUDA extension, used by the item embedding in the unmodified benchmark.
- FlexKV native extension, xxHash, and SSD io_uring support. GDS, P2P, and Prometheus are disabled.
- HSTU attention built for BF16 forward inference and head dimension 256. This is deliberately specific to this benchmark.
- `hstu_cuda_ops`, `paged_kvcache_ops`, and the KV-cache manager extension/runtime library its package builds. The C++ client also requires `cppzmq-dev` for `zmq.hpp`.
- nvCOMP 5.1.0.21: paged KV-cache ops link it even when the benchmark doesn't enable compression.

Skipped: Triton inference server/backends/client build, NVE (both versions), Megatron, TorchX, AOTI executables/plugins, standalone dynamic embedding Torch binding, training attention variants, and unrelated extension `kk_cpu_ops`. `HSTU_INFERENCE_ONLY=1` is set. The Triton **kernel compiler** bundled with PyTorch remains required and is distinct from Triton inference server.

CUDA compilation targets only SM120 for this deployment; HSTU's build uses SM120a. Like upstream, BF16 RTX attention uses its SM89 template specialization compiled for SM120a (`HSTU_DISABLE_120=TRUE`, `HSTU_DISABLE_86OR89=FALSE`); the separate FP8 RTX path is disabled. Do not change these flags based on their names alone.

The NGC base is still a large development image. This minimizes extra source builds, not the absolute image size. A small runtime-only image is a separate packaging exercise.

## Build on the RTX server

After cloning the transfer branch (see the final chat message for its URL):

```bash
cd recsys-examples
bash docker/flexkv-minimal/prepare.sh
BUILD_JOBS=4 bash docker/flexkv-minimal/build.sh sm120 2>&1 | tee build-sm120.log
GPUS=device=2 bash docker/flexkv-minimal/run.sh preflight
```

Use `set -o pipefail` when automating the command piped to `tee`. Lower `BUILD_JOBS` to 2 if compiler processes run out of host RAM. The base needs network access to Ubuntu, GitHub, PyPI, PyTorch wheel hosting, and NVIDIA nvCOMP hosting. Only the selected submodules are initialized; a full recursive clone is unnecessary.

The build uses temporary CUDA/NVML stubs to load/link extensions in Docker's GPU-free build environment. They are removed from the finished filesystem; the actual driver libraries must come from `docker run --gpus ...`.

## Smoke checks, then measurement

Start with a small run without CUDA Graph capture:

```bash
bash docker/flexkv-minimal/run.sh no_cache --warmup-iters 1 --timed-iters 2 --disable-cudagraph
bash docker/flexkv-minimal/run.sh gpu_hit --warmup-iters 1 --timed-iters 2 --disable-cudagraph
bash docker/flexkv-minimal/run.sh cpu_hit --warmup-iters 1 --timed-iters 2 --disable-cudagraph
```

Then use the defaults (history 1024, no appended history, batch size 1, CUDA Graph enabled, warmup setting 100, timed iterations 50):

```bash
bash docker/flexkv-minimal/run.sh no_cache
PROFILE=1 bash docker/flexkv-minimal/run.sh gpu_hit
PROFILE=1 bash docker/flexkv-minimal/run.sh cpu_hit
```

`PROFILE=1` requires `nsys` inside the image. The NGC base normally includes it; check `docker run --rm hstu-flexkv-minimal:sm120 which nsys`. The original upstream GPU/CPU cases expose `scenario1_timed_run_*` and `scenario2_timed_run_*` NVTX ranges rather than printed latency summaries. Inspect those ranges and GPU activity in Nsight Systems; Python range duration alone is not a synchronized GPU latency measurement. The mounted benchmark now prints synchronized wall samples for all cases; Nsight remains useful for a component timeline. Do not report “completed” messages as numerical performance results.

The upstream GPU/CPU preparation loops use the timed-user count rather than `--warmup-iters`. The script passes the upstream flags unchanged. Do not assume 100 extra steady-state warmups happened in those cases.

Baseline and cache scenarios must be separate processes: upstream rejects `no_cache` combined with cache scenarios. Cache-hit runs reserve about 10 GiB for the model's primary GPU KV cache alone; CUDA Graphs, model weights, workspaces, and request tensors use additional memory. FlexKV also allocates pinned host memory. Increasing history, batch size, or timed iterations changes memory pressure. `--disable-cudagraph` changes the measurement mode; compare matching modes.

## SSD tests

Choose a dedicated empty directory on the actual local SSD, not a shared filesystem, tmpfs, or container overlay. Ensure capacity and write permission; the recipe doesn't delete existing user data.

```bash
export SSD_CACHE_DIR=/your/local/ssd/flexkv-benchmark-cache
PROFILE=1 bash docker/flexkv-minimal/run.sh ssd_hit
PROFILE=1 LAYER_GRANULARITY=1 bash docker/flexkv-minimal/run.sh ssd_layerwise
```

The wrapper mounts that directory at `/ssd-cache`, writes a config into `results-flexkv/ssd.yml` (CPU cache 0.2 GiB, SSD cache 8 GiB), and uses `--no-prefetch` for naive/layerwise comparability. The 8 GiB SSD setting accommodates the default targets plus pressure users more safely than upstream's tiny illustrative config. If increasing the workload, reassess capacity and the SSD residency prechecks; a recompute/fallback is not an SSD hit.

Each run saves a log under `results-flexkv/` by default, plus an `.nsys-rep` when profiled. Override `RESULTS_DIR`, `IMAGE`, or `GPUS` as needed. Use separate results directories to preserve repeated measurements. Test granularity 1, 2, 3, 4, and 8 in separate runs. Do not compare H100 timings with RTX timings as if they measured the same hardware. Upstream uses a fail-open host-cache policy: inspect warnings and cache-residency checks before accepting any result.

## Verification status

**Local verification is static only. User-supplied target logs now confirm a successful image build, GPU preflight, and no-cache/GPU-hit/CPU-hit smoke cases with CUDA Graphs off and on; see `PERF_RESULTS.md`. SSD execution and the newly added cache latency instrumentation remain unverified on the target.** The earlier H100 and SM120 build attempts were canceled during FlexKV compilation at the user's request. No complete image or benchmark run was produced locally. H100 behavior is not used as evidence for RTX PRO 5500 behavior.

The static review traces imports and native linkage, checks the target architecture flags against the pinned build scripts, and inspects the supplied target driver/GPU snapshot. A binary audit of the matched FBGEMM wheel found an SM120a cubin in its cumulative-sum library. This establishes that code target is present, not that all wheel operations or the model execute correctly on the RTX.

See `STATIC_REVIEW.md` for evidence and remaining target-side checks. Target execution evidence comes only from user-supplied logs. New collection commands still need target-side verification.

## References

- [Pinned upstream Dockerfile](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/docker/Dockerfile)
- [Pinned benchmark](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py)
- [NVIDIA PyTorch 26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html)
- [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)

## Run and collect results together (no image rebuild)

From this revision, `run.sh` mounts the host benchmark into the existing image. GPU/CPU cache forward loops now print synchronized wall latency samples, and SSD samples are exported too. The model and cache residency logic are unchanged. Each GPU/CPU sample synchronizes before and after the forward call; request construction and explicit GPU eviction stay outside that timer. Background offload completion is not part of the forward latency. This is sequential request latency, not pipelined throughput.

The host needs Python 3 (standard library only). Set the GPU explicitly to avoid accidentally selecting the server's RTX 6000D at index 0.

```bash
git pull --ff-only
export GPUS=device=2
export RESULTS_DIR="$PWD/results-flexkv-measurements"
export PERF_RECORD_DIR="$PWD/perf"

# A short check of the new timing/collection path; no image rebuild required.
bash docker/flexkv-minimal/run.sh gpu_hit --warmup-iters 1 --timed-iters 2

# All commands run AND gather data. Each run appends its results automatically.
for scenario in no_cache gpu_hit cpu_hit; do
  bash docker/flexkv-minimal/run.sh "$scenario" --timed-iters 50 || break
done

# View the cumulative record and spreadsheet-friendly summary.
cat "$PERF_RECORD_DIR/bench-results.md"
cat "$PERF_RECORD_DIR/summary.csv"
```

Every run gets a unique directory below `$RESULTS_DIR/runs/`. It contains the raw log, exact command, GPU/driver snapshot, image ID, source revision, benchmark hash, `samples.csv`, and `result.json`. Optional `PROFILE=1` also saves the Nsight report in that directory. Use profiling as a separate diagnostic run; mark its overhead when comparing measurements.

While a run is active, another terminal can follow samples as they arrive:

```bash
export RESULTS_DIR="$PWD/results-flexkv-measurements"
tail -f "$(cat "$RESULTS_DIR/latest-run.txt")"/*.log
```

The cumulative `perf/bench-results.md`, `perf/summary.csv`, and append-only `perf/runs.jsonl` update after each run, including failed runs. Initial history is seeded with the six smoke cases supplied in this chat; old GPU/CPU cases have **no inferred latency**, only completion and transfer evidence. Each run with latency samples records mean, median, P95, P99, minimum, maximum, and standard deviation. Raw transfer timings are retained separately and may include warmup activity. Cleanup warnings remain visible. Fewer than 30 samples are marked preliminary; even 50 samples need sensible warmup and repeatability before a firm comparison.

To import a saved log later (the run ID must be unique, and retrying the same ID/content is idempotent):

```bash
python3 docker/flexkv-minimal/collect_perf.py \
  --log /path/to/saved/cpu_hit.log --mode cpu_hit \
  --run-id imported-cpu-hit-001 --record-dir "$PWD/perf"
```

`PERF_RESULTS.md` is the checked-in snapshot of the supplied target evidence. The target's `perf/` directory contains the automatically updated working record and is gitignored so future pulls do not conflict. Send back `perf/summary.csv` and relevant `result.json`/logs when updating the shared snapshot. Instrumentation and collection changes were checked using saved logs, AST/shell syntax, and simulated Docker success/failure; no code was built or run on a GPU here.
