# Static review: RTX PRO 5500 FlexKV benchmark

Status: **statically reviewed candidate, not a completed or runtime-validated build**. No H100 result is used to infer RTX behavior. Both attempted image builds were stopped during FlexKV compilation; no benchmark was run. No further compilation is part of this review.

Source: NVIDIA/recsys-examples `d03198c35166e8dae907b6bc453f2fa1112700b0`. Submodule revisions and resolved base image digest are in `source-lock.json`. Source paths and line numbers below refer to that revision (or its pinned submodules).

## Dependency decisions

| Decision | Source evidence | Consequence |
|---|---|---|
| Keep TorchRec and FBGEMM | Benchmark imports `torchrec.sparse.jagged_tensor`; `configs/task_config.py:19` imports `commons.modules.embedding`, which imports the dynamic embedding planner and TorchRec distributed APIs | A tiny replacement jagged-tensor class would not preserve the existing import graph; use the upstream matched pair |
| Keep dynamic embeddings | Benchmark `:224–240` enables dynamic embeddings for the item feature; `modules/inference_embedding.py:23` imports the native-backed package | Build `corelib/dynamicemb`; skipping it would require modifying model behavior or imports |
| Keep HSTU attention | `modules/paged_hstu_infer_layer.py:23` imports `hstu_attn_varlen_func`; benchmark `:156–160` fixes head dimension 256 / BF16 | Build BF16, head_dim=256, forward inference only |
| Keep two commons CUDA extensions | Paged layer `:16` imports `paged_kvcache_ops`; `commons/ops/cuda_ops/JaggedTensorOpFunction.py:3` imports `hstu_cuda_ops`; ranking model requires `strip_cached_tokens` | Use `BUILD_EXT_ONLY=hstu_cuda_ops,paged_kvcache_ops`; omit CPU partitioner |
| Keep nvCOMP | `examples/commons/setup.py:49–70` unconditionally links `nvcomp_static` into the paged cache extension | The unmodified extension still needs nvCOMP even though this benchmark does not request compression |
| Keep KV-cache manager package | Dense module imports its manager and metadata classes; native GPU manager imports `kvcache_cpp` | Use its stock installer. This also builds `kvcache_manager_ops.so`; avoid another standalone CMake build |
| Keep FlexKV + io_uring + ZMQ | Flex adapter imports FlexKV client/config classes; FlexKV setup links `uring` and `xxhash`; KV-cache manager links ZMQ | Retain local SSD support, native logging headers, xxHash, and `cppzmq-dev` for the C++ `zmq.hpp` header (in addition to `libzmq3-dev`). Disable optional GDS/P2P/metrics |
| Skip NVE | `modules/inference_embedding.py:193–194` imports NVE only for the NVEMB backend; `:246–255` defaults to DynamicEmb/TorchRec | No NVE source build needed for the unchanged benchmark |
| Skip Megatron | `modules/inference_dense_module.py:40–46` guards training imports with `HSTU_INFERENCE_ONLY`; `hstu_processor.py:34–43` and `mlp.py:20–36` provide inference fallbacks | Set `HSTU_INFERENCE_ONLY=1` |
| Skip Triton Server, AOTI apps/plugins, TorchX, standalone embedding Torch binding | They are serving/export/training tooling outside the selected benchmark's execution path | Retain only the Triton kernel compiler already supplied with PyTorch |

This is a conservative reduction using the stock package installers, not a claim of the smallest theoretically possible image or compilation graph. Further reductions would require maintaining source patches.

## RTX-specific architecture settings

NVIDIA lists RTX PRO 5500 Blackwell as compute capability **12.0**. It is not the SM100 data-center Blackwell target.

| Component | Setting | Evidence |
|---|---|---|
| FlexKV / commons CUDA extensions / KV-cache manager CUDA extension | `TORCH_CUDA_ARCH_LIST=12.0` | Their installers use PyTorch CUDAExtension; FlexKV setup honors this environment variable |
| Dynamic embedding native extension and eviction fatbins | `DEMB_ARCHS=120` | `corelib/dynamicemb/setup.py:106–125,198–234` builds its own `-gencode` list; setting TORCH_CUDA_ARCH_LIST alone is insufficient |
| HSTU attention | `HSTU_ARCH_LIST=12.0` | Pinned HSTU `setup.py:211–213` emits `compute_120a,sm_120a` |
| HSTU BF16 template dispatch | `HSTU_DISABLE_120=TRUE`, **`HSTU_DISABLE_86OR89=FALSE`** | Same choice as upstream Dockerfile. Ampere-family `static_switch.h:141–153` maps runtime SM120 to template Arch89; `src/generate_kernels.py:51` must therefore generate the Arch89 specialization. The separate RTX FP8 path is unnecessary |
| KV-cache layout | SM12 uses page size **32** | Benchmark `:181–185` selects 128 only for SM major >=10 and <12; do not copy a B200/SM100 configuration |
| Dense projection | `torch_addmm_silu_fwd` | Paged layer `:34–39` explicitly selects this for SM12 |

The FBGEMM 1.8.0+cu130 wheel is copied from upstream's checksum-pinned recipe, coupled with TorchRec v1.8.0-rc1 and NGC PyTorch 26.07. A `cuobjdump --list-elf` inspection of `fbgemm_gpu_sparse_async_cumsum.so` found an `sm_120a.cubin`. This confirms the target exists in that library; it does not prove compatibility of every FBGEMM operation or full-model execution.

## Target environment evidence

User-supplied `nvidia-smi`: driver **620.70.06**, CUDA driver API **13.5**, RTX PRO 5500s at physical indices **2 and 3**. NGC PyTorch 26.07 uses CUDA 13.3.1. The driver is sufficiently recent on paper; container GPU injection, dynamic library resolution, PTX JIT, and application execution still require checking on the target.

Both target RTX GPUs showed ~80,379 MiB used out of 85,651 MiB. Remaining memory was roughly 5 GiB. Primary GPU KV-cache allocation alone is 10 GiB:

`327680 tokens × 2 (K,V) × 1024 elements × 2 bytes × 8 layers = 10 GiB`.

Select physical GPU 2 or 3 explicitly after it is made available. Physical GPU 0 in that snapshot is an RTX 6000D and would measure a different model. No process termination or target-side action was performed.

## Static checks completed

- `bash -n` on all three shell scripts: passed.
- Python AST parse of `preflight.py`: passed, without importing/executing it.
- `docker build --check`: passed, no RUN steps executed and no compilation performed.
- Architecture variables inspected against all selected native installers and HSTU generator/dispatch code.
- Docker context exclusion corrected to preserve FlexKV's required `csrc/dist/lease_meta_mempool.h`. A generic `**/dist` exclusion would silently remove it.
- CUDA/NVML build-only loading paths inspected. Temporary driver stubs are removed from the final filesystem, and must not substitute for actual injected driver libraries at runtime.
- Source and submodule hashes recorded; transfer branch preserves the upstream benchmark/model source.

The recipe intentionally retains FlexKV release/Cython compilation (`FLEXKV_DEBUG=0`). Disabling it could shorten the build, but changes Python execution overhead and is not treated as an equivalent benchmark configuration here. No source build time or image size is claimed from the canceled attempts.

## Checks still required on the RTX server

1. Run the stock 26.07 image with `--gpus device=2` (or 3), confirm the actual name and `(12,0)`, and execute a small BF16 CUDA operation.
2. Build the image there. Static review cannot guarantee compiler success or ABI linkage.
3. Run the included preflight, then separate small no-cache/GPU-hit/CPU-hit smoke cases. Check for missing native ops, unsupported kernel images, and allocation failures.
4. Test CUDA Graph mode separately on the target. Then check SSD permissions, io_uring availability, cache residency, and naive/layerwise behavior on its actual SSD.
5. Only then collect performance measurements with matching shapes and graph modes. Inspect fallback warnings: upstream sets a fail-open cache policy. GPU/CPU timed NVTX ranges require profiling and are not printed latency summaries.

## Primary references

- [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus)
- [NGC PyTorch 26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html)
- [Pinned upstream build](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/docker/Dockerfile)
- [Pinned benchmark](https://github.com/NVIDIA/recsys-examples/blob/d03198c35166e8dae907b6bc453f2fa1112700b0/examples/hstu/inference/benchmark/inference_benchmark_flexkv.py)
- [Pinned attention setup](https://github.com/jiayus-nvidia/FBGEMM/blob/17a613fc8bf7df7b19cd32bb1874c54642821793/fbgemm_gpu/experimental/hstu/setup.py)

## Target build follow-up: missing zmq.hpp

The target build reached `recsys_kvcache_manager` and failed at `flexkv_cpp_client.cpp:12` because `zmq.hpp` was absent. Ubuntu Noble packages the C++ headers in `cppzmq-dev`; `libzmq3-dev` only recommends it, so `--no-install-recommends` did not install it. The Dockerfile now explicitly installs `cppzmq-dev` and checks `/usr/include/zmq.hpp` immediately before the KV-cache manager build. Earlier compilation layers are unchanged to preserve the target's Docker cache. This fix was reviewed statically; no compilation was performed here.

Package evidence: [Ubuntu cppzmq-dev](https://packages.ubuntu.com/en/noble/cppzmq-dev), [Ubuntu libzmq3-dev](https://packages.ubuntu.com/noble/libzmq3-dev).

## Target evidence and measurement follow-up

The user subsequently supplied a successful SM120 image build, a preflight identifying RTX PRO 5500 / capability 12.0 / 82.61 GiB free, and six completed no-cache/GPU-hit/CPU-hit smoke cases (graphs off and on). `PERF_RESULTS.md` and `perf-seed.jsonl` preserve the supplied performance evidence. The two original no-cache runs each contain two synchronized latency samples; the four cache runs prove completion but contain no inference latency samples. Destructor/CUDA IPC cleanup warnings are retained.

A later instrumentation update adds synchronized wall latency around GPU/CPU forwards, per-scored-request SSD samples, and a host-side collector. `run.sh` bind-mounts this benchmark into the built image, preserving existing compiled extensions. This intentionally changes benchmark timing code; the initial review's statement about untouched benchmark source applies only to the earlier recipe. Cache timing synchronizes each iteration, changing scheduling relative to the upstream asynchronous loop. It reports sequential forward latency, excludes explicit eviction/request construction, and does not wait for background offload completion. No new GPU execution or compilation was performed locally.

## Longer target run: baseline captured, profiled cache preparation failed

A further user transcript using the earlier launcher contains 50 graph-enabled no-cache latency samples (mean 2.5143 ms) and a profiled GPU-hit attempt that failed during preparation at `FlexKV GlobalCacheEngine._check_input`: `slot_mapping.size=0, token_mask.sum()=2048`. No GPU-hit timed samples were collected; no CPU-hit execution appears in the supplied transcript. The record preserves the successful baseline separately from the failure. No performance comparison is valid yet.

The source shows the slot mapping is built from GPU page-table metadata during onboard launch and reused for PUT. The assertion establishes an empty mapping against a nonempty history, but this transcript alone does not establish why the mapping became empty. The attempt used Nsight profiling and the older benchmark/launcher; retry with the current collector without profiling to distinguish reproducibility from profiler-related behavior. No claim is made that timing instrumentation fixes this assertion. Local checks remain static/log parsing only.

## Unprofiled round 1: mapping failure reproduced

The target subsequently ran the current collector with `PROFILE=0`. No-cache completed 50 samples (mean 2.507902 ms, P95 2.5178 ms, exit 0). GPU-hit again failed during preparation with the same empty-slot assertion (exit 1, zero timed samples). CPU-hit was not reached because the loop stopped. This establishes that profiling is not required to reproduce the failure. The collector correctly saved both results.

A benchmark-side invariant check now detects a nonempty history paired with an empty slot mapping immediately after onboard launch, captures request/user IDs, page-indirection contents, buffer lengths, cache lookup lengths, and a post-synchronization re-read/rebuild, then fails before PUT. It does not substitute a mapping, retry the request, or bypass the assertion. Additional GPU reads/synchronization occur only on the already-invalid path. No claim is made yet about the cause. A graph-disabled run is the next diagnostic comparison; it is recorded separately from the graph-enabled round. No native extension rebuild is needed for this evidence check.


## Graph-disabled failure and allocator correction

The target diagnostic failed on allocation 19, user 28, with 64 page indices but offsets `[1007943680, 0]` for 2048 tokens. Explicit synchronization and rebuilding the mapping did not repair it. Eighteen preparation transfers completed; zero inference samples were measured. Profiling and CUDA Graphs were disabled. The eleventh entry in the shared result record preserves this evidence.

Static inspection identifies a missing incoming stream dependency in `GPUKVCacheManagerImpl::allocate`. Python allocates buffers on the current stream; native CUDA writes them on a separate nonblocking stream. The original outgoing event wait cannot protect recycled storage against earlier pending writes. [PyTorch documents this requirement even for writes into empty tensors](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-streams). This is a source defect and the leading causal explanation, still awaiting confirmation on the target.

The patch adds an incoming event, retains the outgoing event, and protects CPU reuse/freeing of the single pinned staging buffer by waiting for its prior completion. The native module exposes an `events_v1` provenance marker. A separate Docker recipe rebuilds only the cache-manager package atop the already-built image. A target-only regression queues preceding writes to the metadata buffers on default/non-default streams; it has not been executed locally.

The opt-in `--sync-cache-alloc` diagnostic uses a current-stream CPU wait with the existing image. It changes latency and is marked in config, CSV and Markdown. Invalid metadata still fails; there is no substitution or retry. Local verification remains static/source review, Python/shell syntax, saved-log parsing, and CPU-only mocks. Target runtime confirmation is pending.


## Allocation synchronization diagnostic passed on the RTX target

Run `gpu_hit-20261110T093804Z-eX4F8j` used the original image (`allocator_impl=unmarked`) with `--sync-cache-alloc`, CUDA Graphs disabled, and `PROFILE=0`. All 50 preparation users passed the GPU-residency precheck at 2048 tokens each. Fifty timed forwards completed, exit code 0; no cache metadata errors were reported. The CUDA IPC cleanup warning remains recorded after successful completion.

The saved target JSON and all 50 printed samples agree: mean 3.06805244 ms, median 2.96755 ms, P95 3.0195955 ms, minimum 2.955089 ms, maximum 7.734595 ms. The maximum is the first timed sample; all samples remain in the record. This is diagnostic timing with an extra allocation synchronization and graphs disabled. It must not be directly compared to the prior graph-enabled no-cache baseline or treated as performance of the native correction. The upstream GPU-hit loop prepares the prefixes but does not perform the requested count of extra GPU-hit warmup forwards.

Preventing pending caller-stream work before allocation removed the failure in this run. This strongly supports the ordering-race diagnosis, but the broad wait could also mask other timing defects. The event-based native correction is not yet runtime validated: the image still contains the original extension. Next, build only the cache-manager package on the target, run its focused stream-ordering regression, and repeat the same graph-disabled 50-request case without `--sync-cache-alloc`. Accept that validation only with `allocator_impl=events_v1`, `sync_cache_alloc=False`, the expected GPU prefix precheck, 50 samples, and exit 0. No local compilation or GPU execution was performed.


## Native correction built; test launcher lacked pytest

The user-provided validation log `validation-1SfHJj` confirms the cache-manager package built and installed successfully. The Docker RUN step took 32.4 seconds, including the successful native marker check `cache_allocator_stream_ordering == events_v1`. The resulting image is `hstu-flexkv-minimal:sm120-allocfix`, SHA256 `a5d4d98ea8fb979b9c03b0fe13e67b8e0e76d880f24f66ff83bcda966b772f5c`. This is target build/import evidence, not GPU execution of the corrected allocator.

The regression launcher then exited with `/usr/bin/python: No module named pytest`. Neither regression test started, and the shell's failure handling prevented the benchmark from running. The earlier instructions assumed pytest was available; that assumption was incorrect for the minimal image. No cache-allocator runtime failure or new benchmark samples were observed in this attempt.

The regression now uses standard-library `unittest` with two named stream tests, explicit native-marker verification, and an error if CUDA is absent. The updated test is bind-mounted into the existing corrected image, so neither a package installation nor a rebuild is required. Only Python/shell syntax and CPU-only runner mocks are verified locally; target runtime validation remains pending.
