# hstu-poc: HSTU inference with remote KV over RDMA (POC)

**Status:** design only. No code yet; P0 and P1 run together later. See the plan in the study doc §10.
**Design and rationale:** `/lustre/fsw/coreai_devtech_all/junyiq/hstu-kv-rdma/docs/hstu-kv-rdma-study.md`

**Goal:** the fastest achievable HSTU inference when user-history KV lives in remote memory, using plain kernels and calls with no KV-manager layer.

**What it compares:**
- four transports: **CPU verbs**, **GIN-Proxy**, **GIN-GDAKI**, **raw GDAKI** (DOCA GPUNetIO without NCCL)
- against **recompute**, a **local GPU-memory hit** and **local host H2D**

## Scope
- **Workloads:**
  - **R:** history fully cached remotely, C candidates, no new tokens.
  - **W:** prefix P ≥ 0 already local (P = 0 = fresh sequence), S new tokens computed and written, C ≥ 0 candidates.
  - **R+W** comes later.
- **Local GPU KV, built in order, non-paged first:**
  - **A:** no local cache; per-request K/V buffers.
  - **B:** fixed max-length slot per user.
  - **C:** paged pool (later).
- **Attention:** FBGEMM's C++ Hopper kernel for non-paged KV (copied here for the fusion experiments); the paged CuTe-DSL kernel for Mode C.
  - **History prefix:** queries shorter than keys plus targets is undocumented; test it against `pytorch_cached_hstu_mha`.
  - **Fallback:** two calls summed, which is exact because HSTU attention has no softmax.
- **Read timing (5′a):** a post kernel at request start, and a wait kernel before each layer's attention.
- **Fusion experiments (P4, before the serving layer):** compared against CUDA graphs and PDL; the best variant goes into the serving tests (P5).
  - **(i)** wait at the start of attention
  - **(ii)** wait right before each KV tile's TMA load
- **Write timing (6′):** a pack kernel copies the new tokens' K/V out of the uvqk buffer and issues the put per layer; commit after the last layer.

## Planned layout
```
hstu-poc/
  README.md
  model/      HSTU layer + block (trimmed copy of examples/hstu/modules/paged_hstu_infer_layer.py),
              random weights, CUDA-graph capture per (batch, length) bucket
  attn/       copy of FBGEMM's C++ Hopper HSTU forward kernel (for the fusion experiments)
  kv/         local KV modes A/B (non-paged) then C (paged); ncclMemAlloc single-segment buffers;
              remote host pool (cuMemCreate HOST_NUMA); static index + commit word
  transport/  one interface; cpu_verbs (C++, ibverbs); gin (CUDA C++: post/wait kernels, pack+put),
              GIN-Proxy / GIN-GDAKI via NCCL_GIN_TYPE; raw_gdaki (DOCA GPUNetIO, NCCL's bundled copy)
  bench/      workloads R and W, baselines (recompute, local hit, local H2D), the grid sweeps
              (L x C x B, uniform and skewed batches, 1 and 8 GPUs), data checks
  serving/    thin serving layer (no Triton): open-loop load generator, request queue, dynamic batcher,
              KV fetch at admission, CUDA-graph execution per bucket, per-request timestamps,
              max QPS at a p99 target
  scripts/    Eos launch: srun + pyxis container, env (pattern: hstu-kv-rdma/tools/eos-rdma-smoke.sh)
```

**Objective of the comparison:** the highest throughput at a fixed p99 end-to-end latency, with per-request latency alongside. The baseline grid comes first; the GPU-SM vs CPU-core balance of transfer work is a later phase (P6).

## Design details (agreed; study doc §13–§14)
- **Model configs:** S = 3 layers × 512 (4 × 128 heads), M = 8 × 1024 (4 × 256), D = 16 × 1024 (8 × 128). All BF16, head MLP [512, 8].
- **Embedding:** two GPU-memory tables (items, actions) + an input linear; a bypass option for transfer-only tests; host-memory tables later as a realism check.
- **Pre/post:** interleave items and actions, contextual tokens, linear to hidden, bucketed position encoding, input layer norm. Cached tokens are skipped. Post: output layer norm → candidates → MLP → sigmoid.
- **Inputs:** jagged tensors (values + per-user offsets); no `HSTUBatch`.
- **Cache bookkeeping:** naive and always-hit, but distinct per user. N warm users, each with a seeded slot in the remote pool (with checksums); distinct user IDs per batch from a uniform or Zipf distribution; W appends into the user's slot.
- **Load generator:** homogeneous streams first (fixed L and C per run), then mixed (Zipf α = 1.5, lognormal). Results are latency–throughput curves (p50/p99 vs offered QPS).
- **Remote store:** may have no GPU. CPU verbs and raw GDAKI also run against a GPU-free store process.
- **Code:** new branches pushed to `github.com/geoffreyQiu/recsys-examples`; run `git fetch --unshallow` before the first push.

## Dependencies
- **Container, first try:** `gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:devel_latest`, prebuilt from this repo's Dockerfile. The enroot credentials include gitlab-master; whether a compute node can pull the image is tested at the start of P1.
- **Container, otherwise:** `/lustre/fsw/coreai_devtech_all/junyiq/images/pytorch-26.07.sqsh` plus a reduced FBGEMM HSTU build (sm_90, forward only, only the masks we use).
- **Either way:** export the finished environment as a `.sqsh` on Lustre.
- **HSTU attention:** FBGEMM at `d6249db` (`hstu-kv-rdma/repos/FBGEMM/fbgemm_gpu/experimental/hstu`), built for sm_90. Fetch its CUTLASS submodule when P1 starts.
- **NCCL:** the source build at `/lustre/fsw/coreai_devtech_all/junyiq/src/nccl/build`, which has the GIN device API (header-only; link `-lnccl`). Raw GDAKI uses its bundled `src/transport/net_ib/gdaki/doca-gpunetio/`.
- **Eos specifics:** `OMPI_MCA_coll=^ucc` for MPI; set NCCL variables inside the container (`run.sh`).
