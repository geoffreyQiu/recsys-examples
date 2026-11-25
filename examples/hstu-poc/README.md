# hstu-poc: HSTU inference with remote KV over RDMA (POC)

**Status (2026-11-25):** P0–P6 built and measured on Eos; final report in `docs/results.md` (copy of `/lustre/fsw/coreai_devtech_all/junyiq/hstu-kv-rdma/docs/results.md`); plan in the study doc §10. P7 (hybrid fetch + recompute) not started.
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

## Layout (as built)
```
hstu-poc/
  README.md
  csrc/          C++/CUDA extension (JIT-built by hstu_poc/ext.py into build/ext*)
    gin_kv.*       GIN transport (NCCL 2.32.7 device API): split-phase gets over N contexts, puts, fence, pollers
    verbs_kv.*     CPU verbs: RC QPs, worker thread (optionally spinning; optional inline posting), per-layer
                   host-mapped flags, tiled reads with per-layer chunk progress, gated writes, GPU-less store side
    raw_gdaki.*    raw GDAKI: DOCA GPUNetIO (NCCL's bundled BSD-3 copy, build/doca/libhpoc_doca.so) GPU-side QPs
    kvutil.*       data pattern fill/check, spin/stamp/seq/flag kernels, SM hog, green-context streams
    bindings.cpp   pybind11 module hpoc_ext
  attn/          copy of FBGEMM's Hopper HSTU forward kernel with the P4 wait hook (op namespace hpoc_attn)
  hstu_poc/
    model.py       HSTU layers (repo math), forward_full (recompute) and forward_cached (R and W)
    kvsource.py    KV sources: local hit, host H2D, GIN, CPU verbs, raw GDAKI; latency options (GDAKI zero-copy
                   descriptor and side-branch posting, GIN adaptive contexts) and P4 wait variants (wait kernel, pdl,
                   pdl_early = trigger-and-exit + PDL attention spinning on a ready flag, fused, tile = per-tile waits
                   for CPU verbs: chunked reads posted from the history's end, per-K-tile wait in the attention)
    writers.py     workload W writers (GIN, CPU verbs) with a durability fence
    serving2.py    overlap-optimized servers (P5s/P6s): fetch posts each request into a ring of KV slots on arrival,
                   batches landed requests, one graph per batch size re-pointed at any run of slots; recompute with
                   dynamic batching or one at a time; co-located recompute job (stream / green context)
    serving.py     batch-at-a-time serving layer (two ping-pong regions; P5b and the 8-GPU runs)
    serving_v1.py  the first-pass loop, kept only to reproduce the serving crash (Xid 31)
    attn_ext.py    build/launch of the copied attention kernel
    ext.py, dist.py
  bench/
    p0f_tickets.py  P0-f split-phase across kernels (data checks, negative control, commit into host memory)
    p1_check.py     P1 kernel/model correctness
    grid.py         baseline/transport grid runner (one CUDA graph per cell; --clock host = end-to-end host timing)
    p2.py p3.py     two-node drivers for workloads R and W (--transport gin|raw|verbs; p2: --p4 variant, --tuned or
                    the single latency options --zc --side-post --adapt-ctx --spin --inline-post)
    p4_check.py     slow-producer correctness of the wait variants
    p5s.py          serving capacity per strategy (serving2; --p4 fused, --tuned, --stop sat for relaxed targets up to
                    saturation; --cotenant/--green/--priority for the shared-GPU runs)
    p5.py           latency vs offered load of the batch-at-a-time layer (-n 8 = all GPUs at once)
    p5_v1.py        first-pass driver (serving_v1), crash reproduction only
    p6.py           transfer under SM contention / CPU contention
  scripts/srun2.sh  2-node launcher (inference ranks on node A, store ranks on node B)
  tools/           analysis and campaign scripts behind docs/results.md (plots need matplotlib): make_figs.sh
                   (every figure), p2t_report.py, p2t_to_p2c.py, report_figs.py, p2c_breakeven.py, p5s_report.py,
                   p5x_report.py, p6s_report.py, p3_report.py, p5_eight.py, campaigns (p2t_grid.sh, p5s_*.sh,
                   p5x_runs.sh, p6s_*.sh, ...), numa_check.cu
  docs/            results.md (final report), figs/, archive/ (earlier revisions)
```

## Running (Eos; inside `pytorch-26.07.sqsh`)
```
tools/build-hstu-attn.sh        # reduced FBGEMM HSTU build (task folder), ~2 min
tools/build-doca.sh             # standalone DOCA GPUNetIO for raw GDAKI (task folder), ~1 min
scripts/srun2.sh -j JOB -n 1 -e NCCL_GIN_TYPE=3 -e NCCL_RMA_DISABLE=1 -- python bench/p2.py --transport gin --label gdaki \
    --modes gdaki,local,h2d --configs M --L 4096,16384 --B 1,8 --C 256 --isolated --out results/p2/x
```
Always set `NCCL_RMA_DISABLE=1` for GIN (otherwise NCCL's host-RMA progress thread spins a core).

**Objective of the comparison:** the highest throughput at a fixed p99 end-to-end latency, with per-request latency alongside. The baseline grid comes first; the GPU-SM vs CPU-core balance of transfer work is a later phase (P6).

## Design details (agreed; study doc §13–§14)
- **Model configs:** S = 3 layers × 512 (4 × 128 heads), M = 8 × 1024 (4 × 256), L = 8 × 2048 (8 × 256); D = 16 × 1024 (8 × 128) is parked. All BF16, head MLP [512, 8].
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
