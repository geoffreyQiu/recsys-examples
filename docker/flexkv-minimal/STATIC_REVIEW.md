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
| Keep FlexKV + io_uring + ZMQ | Flex adapter imports FlexKV client/config classes; FlexKV setup links `uring` and `xxhash`; KV-cache manager links ZMQ | Retain local SSD support, native logging headers, and xxHash. Disable optional GDS/P2P/metrics |
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
