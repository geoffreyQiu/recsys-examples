# Recsys KVCache Manager

Recsys KVCache Manager is a Python package that provides LLM-compatible KV data caching, storage, and lookup for generative recommender model inference.
It supports KV-cache management based on recommender-system **user IDs**, enabling KV-cache reuse across requests from the same user.

Recsys KVCache Manager is based on the **PyTorch** ecosystem. It manages KV data in GPU memory and host memory backed by lower-tier storage.
It supports lookup, offloading to lower-tier storage, onboarding to GPU memory for inference, and read/write APIs for KV-cache data.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Basic APIs](#basic-apis)
- [FlexKV Backend Options](#flexkv-backend-options)
- [C++ Cache Owner](#c-cache-owner)
- [Important Notes](#important-notes)
- [Examples](#example)
- [Future Plans](#future-plans)

## Features

- **User ID Based Caching**:
For recommender systems, behavior sequences that generate KV data vary greatly across users, and there is usually little common prefix between input sequences.
Recsys KVCache Manager supports data lookup by user ID instead of comparing sequence token values.

<div style="margin-left: 2em;"> 

**Recsys KVCache Manager** modules:

</div>

```
    + KVCacheManager :  Interface for KV-cache operations
    |
    ├---- GPUKVCacheManager :  Manager for GPU KV-cache table
    |
    └---- HostKVStorageManagerBase :  Interface of manager for host memory/SSD/remote KV cache
        |
        ├---- NativeHostKVCacheManager :  Wrapper for pinned-host-memory-only KV cache
        |
        └---- FlexKVCacheManager :  Wrapper to FlexKV cache system
```

- **Paged GPU KVCache Table**:
The GPU KV-cache table is organized as a paged KV-data table and supports KV data add/append, lookup, and eviction. When appending new data to the GPU cache, it evicts data from the oldest users by LRU policy if there is no empty page. The HSTU attention kernel from FBGEMM-HSTU can load KV data directly from a paged table, avoiding additional data copies.

- **Asynchronous Onboarding/Offloading**:
By using asynchronous data copy on the side CUDA stream, we overlap the KV data transfer between GPU memory and host storage (onboarding/offloading) with embedding lookup, sequence pre-/post-processing, and inference for other requests (in some cases) to reduce the latency of HSTU inference.
Furthermore, the `NativeHostKVCacheManager` backend supports layerwise KV data onboarding, overlapping the H2D data transfer with computation from the previous HSTU layers.

- **Extension for Multiple Backend**: 
`HostKVStorageManagerBase` is provided as an interface for other LLM-compatible KV-cache systems for host memory, storage, and remote data pools.
This can be extended to integrate other KV-cache systems. Currently, we provide integration with [`FlexKV`](https://github.com/taco-project/FlexKV/tree/main) as the lower-tier KV storage backend.

- **[NEW] Compatible with Torch Export and AOTI**: 
Recsys KVCache Manager now includes an export-compatible backend for KV-cache lookup, allocation, onboarding, offloading, and eviction when using the paged GPU KV cache together with the FlexKV host cache backend.
The export path keeps the Python API unchanged, but routes KV-cache operations through tensor-only torch custom ops while keeping stateful cache policy and runtime management in C++, which makes the flow compatible with `torch.export` and AOTInductor/AOTI packaging. The C++ runtime requires the binary-protocol adapter in the [pinned FlexKV submodule](../../third_party/FlexKV). Preserve this adapter when upgrading FlexKV; the upstream Python pickle protocol alone cannot serve the C++ client. The adapter is based on the current layerwise-capable FlexKV revision and retains the Python protocol.



## Installation

To install, please use the following command:

```bash
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" pip3 install --no-build-isolation .
```

**Note**: To enable the FlexKV backend, install the FlexKV package according to the [FlexKV doc](https://github.com/taco-project/FlexKV/tree/main#how-to-use).


## Basic APIs

-   `lookup_kvcache`: gets the cached sequence length from both the GPU cache table and host KV storage.

-   `allocate_kvcache`: assigns required cache pages in the GPU cache table for inference.
This generates page IDs for reading/writing data and metadata for appending KV data into the GPU table.
It also evicts used cache pages when running out of empty pages. **No offloading** is performed upon eviction in the current implementation.

-   `onboard_launch`: launches async host-to-GPU KV-cache transfer.

-   `onboard_try_wait`, `onboard_wait`: performs non-blocking/blocking waiting for KV data onboarding.

-   `offload_launch`: launches async GPU-to-host offload and records the task into ongoing offload queue.

-   `offload_try_wait`: polls ongoing offload tasks, finishes ready tasks, cancels failed or timed-out tasks, and unlocks host KV storage state.

-   `evict`, `evict_all`: explicitly evicts cached data from the GPU cache table and/or lower-tier storage if supported.

## FlexKV Backend Options

When `host_kvstorage_backend` is set to `flexkv`, additional FlexKV options can be passed through `extra_configs`.

| Option | Default | Description |
| --- | --- | --- |
| `flexkv_as_batch` | `1` | Enables batched FlexKV launch for multi-user onboard/offload requests. Set to `0` / `false` to disable. |
| `flexkv_config_path` | `None` | Optional FlexKV tier config path. Use this to enable SSD-backed cache tiers or other FlexKV user config settings. |

Example:

```python
extra_configs = {
    "flexkv_mode": "direct",
    "flexkv_as_batch": 1,
    "flexkv_config_path": "corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml",
}
```

The sample SSD config is:

```yaml
cpu_cache_gb: 0.2
ssd_cache_gb: 0.5
ssd_cache_dir: /path/to/your/ssd_cache
enable_gds: false
enable_p2p_cpu: false
enable_p2p_ssd: false
enable_3rd_remote: false
```

Before using SSD cache, update `ssd_cache_dir` to a writable local SSD path and set `ssd_cache_gb` > 0.

## C++ Cache Owner

Set `owner_loop: 1` in the YAML selected by `KVCACHE_MANAGER_CONFIG_FILE`
to use the shared C++ owner with the tensor-only `kvcache_manager_ops` API.
`owner_loop: 0` selects the legacy control path. Both use the same lookup,
allocation, onboard and offload operator signatures. Graphs that have no
request-completion boundary should be re-exported with `request_finish` as
shown below before using paged owner execution.
The legacy path blocks on the producer CUDA stream before offload publication;
the owner path uses its recorded completion event instead.
Legacy callers must also reap/flush completed offloads before reusing pinned
pages under allocation pressure; the owner performs this drain automatically.
Both paths drain pending offloads on shutdown. The test's `--legacy` mode
records its explicit allocation-pressure flush separately from the owner's
automatic progress check.

`onboard_child_tasks: 1` selects child-task completion/cleanup for the pinned
FlexKV adapter that retains merged child records. The default `0` preserves
aggregate-task completion for newer adapters. The HSTU startup helpers select
the pinned-adapter policy when its older configuration API is detected;
an explicit YAML value overrides that compatibility default.
For retained-child servers, polling requires terminal completion before
reaping child records; an early readiness response alone does not clean them up.

The owner serializes cache metadata access, dispatches FlexKV transfers, and
progresses completion on its worker thread. `onboard_launch` queues work;
`onboard_wait` remains the required dependency before consuming restored K/V.
`offload_launch` records the producer CUDA stream, so publication cannot read
unfinished K/V. `offload_reap_completed` collects completed publications and
`offload_flush` drains them. Shutdown drains outstanding work before releasing
cache storage; stop the FlexKV server **after** `shutdown_runtime`.

Finish each request with `result = torch.ops.kvcache_manager_ops.request_finish(result)`.
This op returns a fresh clone of the output and records CUDA completion after
the clone. Returning that tensor keeps the completion call live through
`torch.export` and AOTInductor without changing the public output structure.
If the graph launches offload, pass its ticket as the optional second input:
`request_finish(result, offload_ticket)`. This dependency keeps publication
live without exposing the ticket as a model output.
Paged allocation/eviction protects pages until the previous request and its
transfers complete. This is conservative lifetime protection for the existing
single-instance execution model, not a concurrent multi-request allocator.
Use the model's consuming CUDA stream and do not capture these host-driven
standard operators in a CUDA graph.

Specialized layout adapters can link the exported `CacheOwner` API from
`src/runtime/cache_owner.h` (installed under `include/recsys_kvcache_manager`).
The fixed-history converter uses this same worker directly and keeps its
resident-hit metadata and CUDA-graph replay adapters; it does not create a
second owner queue.

Python AOTI can hold the GIL while calling native operators. `OwnerWaitGuard`
releases an already-held GIL around blocking owner waits, allowing tensor
reference cleanup on the worker. It resolves optional CPython symbols at
runtime and is a no-op in Python-free C++ serving. A focused CPU-only regression
can be run from the repository root:

```bash
python corelib/recsys_kvcache_manager/test/test_owner_wait_guard.py \
  --output /tmp/recsys-owner-wait-guard
```

The HSTU [AOTI example](../../examples/hstu/inference_aoti/README.md) enables
the owner and queues history publication by default. The following small
test exercises GPU → FlexKV CPU → GPU without a checkpoint:

```bash
python corelib/recsys_kvcache_manager/test/test_owner_runtime.py \
  --library corelib/recsys_kvcache_manager/build/kvcache_manager_ops.so \
  --output /tmp/recsys-owner-validation --aoti
```

Run from the repository root with the manager and pinned FlexKV adapter
installed. The test uses two layers, 32-token pages, twelve GPU pages, and real
FlexKV CPU storage. It checks partial-prefix append, GPU/CPU/miss mixtures,
producer-stream ordering under allocation pressure, error recovery, shutdown,
and small native-op AOTI graphs. `result.json` lists the checks; `read_graph.py`
and the small PT2 files permit inspection of retained calls. The test is
validation, not a performance benchmark. Cache writes and model execution run
under `torch.inference_mode()`, matching the owner-allocated inference tensors
and the HSTU inference path.

## Important Notes

There are some **limitations** in the current implementation. These are expected to be resolved for broader use cases and better performance.

1. API `allocate_kvcache` is host blocking, and cannot overlap with other operations.

2. Only **one** GPU KV-cache manager is allowed per device, and only one inference instance is allowed for each GPU KV-cache manager.

3. The `native` host backend is limited to at most **one** GPU KV-cache manager and one inference instance. Use it with `user_id`-based routing and inference instance isolation.


## Example

**Typical Usage of KVCache Manager**:
```python
    # Input User IDs
    user_ids: torch.Tensor
    # Sequence lengths per user
    sequence_lengths: torch.Tensor

    # Lookup 
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)

    # strip cached tokens from input sequences
    [[ ... strip cached tokens ... ]]

    # Allocate in GPU cache table
    kvcache_mgr.offload_try_wait()  # [Optional] Try to free up GPU cache space by completing offloading.
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    # Onboard to GPU cache table, non-blocking, if necessary
    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)

    [[ ... embedding lookup ... ]]
    [[ ... preprocess ... ]]


    # Dense Module computation
    # Note: Here we show two possible ways to synchronize with onboard completion:
    #       (1) blocking wait for onboard completion, for non-layerwise backends
    #       (2) non-blocking CUDA stream wait with layerwise onboard events

    # Case (1): Blocking total wait. [ Note: not supported with "native" backend. ]
    kvcache_mgr.onboard_wait(index_meta, kvcache_metadata.kv_onload_handle)
    for layer_idx in range(num_layers):

        [[ ... write new KV data through `kvcache_metadata.kv_cache_table` ]]  # See `kvcache_mgr.gpu_kvcache_mgr.put`

        # Case (2): Layerwise stream wait. [ Note: only active with layerwise backends such as "native". ]
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

        [[ ... attention computation, loading data using `kv_indices`, `kvkv_indptr`, etc. ... ]]
        

    # Offloading to host and lower tiers, non-blocking, if necessary
    kvcache_mgr.offload_try_wait() # [Optional] Try to free up host buffer/sync host caching status by completing offloading.
    kvcache_mgr.offload_launch(index_meta)

    [[ ... postprocessing ... ]]
    [[ ... return inference results ... ]]
```

**Refer to** HSTU model inference in RecSys Examples for details: [InferenceRankingGR](../../examples/hstu/model/inference_ranking_gr.py), [InferenceDenseModule](../../examples/hstu/modules/inference_dense_module.py).


## Future Plans

1. Support concurrent KV-cache operations across inference instances.
2. Broaden Torch Export and AOTInductor coverage for recommender-model KV-cache inference paths in the Torch C++ runtime.
