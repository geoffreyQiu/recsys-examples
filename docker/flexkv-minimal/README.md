# Minimal HSTU FlexKV benchmark build

This recipe runs `examples/hstu/inference/benchmark/inference_benchmark_flexkv.py` from upstream revision `d03198c35166e8dae907b6bc453f2fa1112700b0`. It preserves the benchmark and model source. Default deployment target: NVIDIA RTX PRO 5500 Blackwell, compute capability 12.0, on Linux x86_64. Build on the deployment server; transfer source through GitHub rather than copying H100-only binaries.

## Chosen stack and scope

Base: `nvcr.io/nvidia/pytorch:26.07-py3` (Python 3.12, PyTorch 2.13 development build, CUDA 13.3.1), matching the upstream Dockerfile. The successful pull of 26.03 establishes registry access; it does not validate this stack or the GPU driver. Your supplied target snapshot shows driver 620.70.06 / CUDA driver API 13.5, sufficiently recent for this stack. Its RTX PRO 5500s are physical GPUs 2 and 3; both were occupied (~80 GiB each by PID 691274). Arrange for one to be available before running the benchmark. GPU 0 is an RTX 6000D, so the default GPU 0 would benchmark the wrong model.

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
- `hstu_cuda_ops`, `paged_kvcache_ops`, and the KV-cache manager extension/runtime library its package builds.
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

`PROFILE=1` requires `nsys` inside the image. The NGC base normally includes it; check `docker run --rm hstu-flexkv-minimal:sm120 which nsys`. GPU/CPU cases in this upstream revision expose `scenario1_timed_run_*` and `scenario2_timed_run_*` NVTX ranges rather than printed latency summaries. Inspect those ranges and GPU activity in Nsight Systems; Python range duration alone is not a synchronized GPU latency measurement. `no_cache` prints synchronized wall latency samples. Do not report “completed” messages as numerical performance results.

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

**Static review only; target build and execution are unverified.** The earlier H100 and SM120 build attempts were canceled during FlexKV compilation at the user's request. No complete minimal image or benchmark run was produced. H100 behavior is not used as evidence for RTX PRO 5500 behavior.

The static review traces imports and native linkage, checks the target architecture flags against the pinned build scripts, and inspects the supplied target driver/GPU snapshot. A binary audit of the matched FBGEMM wheel found an SM120a cubin in its cumulative-sum library. This establishes that code target is present, not that all wheel operations or the model execute correctly on the RTX.

See `STATIC_REVIEW.md` for evidence and remaining target-side checks. The commands in this guide are instructions for the target operator; they have not been run on that server.

## References

- [Pinned upstream Dockerfile](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/docker/Dockerfile)
- [Pinned benchmark](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py)
- [NVIDIA PyTorch 26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html)
- [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
