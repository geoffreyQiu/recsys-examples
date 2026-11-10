# Running benchmark results — RTX PRO 5500

Updated automatically from saved logs. UTC timestamps identify collection, not necessarily execution.

Forward latency includes the model call and its completion on CUDA; request construction and cache eviction are outside the timer. Each iteration is synchronized, so this is sequential latency, not pipelined throughput. Transfer timings are separate components. Profiling can add overhead. Fewer than 30 samples are preliminary; more samples alone do not establish steady state or correctness.

| Run | Case | Graphs | Profiled | N | Mean ms | P50 ms | P95 ms | Min–max ms | Status |
|---|---|---|---|---:|---:|---:|---:|---|---|
| reported-smoke-graphs-off-no_cache | no_cache | off | no | 2 | 3.6346 | 3.6346 | 4.3664 | 2.8215–4.4477 | completed |
| reported-smoke-graphs-off-gpu_hit | gpu_hit | off | no | 0 | — | — | — | — | completed; warnings |
| reported-smoke-graphs-off-cpu_hit | cpu_hit | off | no | 0 | — | — | — | — | completed; warnings |
| reported-smoke-graphs-on-no_cache | no_cache | on | no | 2 | 3.3598 | 3.3598 | 3.9270 | 2.7297–3.9900 | completed |
| reported-smoke-graphs-on-gpu_hit | gpu_hit | on | no | 0 | — | — | — | — | completed; warnings |
| reported-smoke-graphs-on-cpu_hit | cpu_hit | on | no | 0 | — | — | — | — | completed; warnings |
| reported-long-graphs-on-no_cache | no_cache | on | no | 50 | 2.5143 | 2.5144 | 2.5267 | 2.4955–2.5339 | completed |
| reported-long-graphs-on-gpu_hit | gpu_hit | on | yes | 0 | — | — | — | — | failed_or_incomplete; warnings |
| no_cache-20261110T085850Z-RUtt2Q | no_cache | on | no | 50 | 2.5079 | 2.5070 | 2.5178 | 2.4862–2.5595 | completed |
| gpu_hit-20261110T085900Z-bBzka0 | gpu_hit | on | no | 0 | — | — | — | — | failed_or_incomplete; warnings |
| gpu_hit-20261110T090542Z-unFSqH | gpu_hit | off | no | 0 | — | — | — | — | failed_or_incomplete; warnings |
| gpu_hit-20261110T093804Z-eX4F8j | gpu_hit | off | no | 50 | 3.0681 | 2.9676 | 3.0196 | 2.9551–7.7346 | completed; warnings; alloc-sync diagnostic |
| allocator-validation-1SfHJj | allocator_validation | ? | no | 0 | — | — | — | — | build_completed_validation_blocked; warnings |

## Run details

### reported-smoke-graphs-off-no_cache

- Assessment: preliminary (<30 samples).
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-off)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "True", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "no_cache", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 0; see JSONL/run JSON for individual values. These may include warmup transfers.

### reported-smoke-graphs-off-gpu_hit

- Assessment: completion only; latency unavailable.
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-off)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "True", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 2; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### reported-smoke-graphs-off-cpu_hit

- Assessment: completion only; latency unavailable.
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-off)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "True", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "cpu_hit", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 4; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### reported-smoke-graphs-on-no_cache

- Assessment: preliminary (<30 samples).
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-on)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "no_cache", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 0; see JSONL/run JSON for individual values. These may include warmup transfers.

### reported-smoke-graphs-on-gpu_hit

- Assessment: completion only; latency unavailable.
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-on)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 2; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### reported-smoke-graphs-on-cpu_hit

- Assessment: completion only; latency unavailable.
- Source log: `user-provided RTX PRO 5500 smoke transcript (graphs-on)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "cpu_hit", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 4; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### reported-long-graphs-on-no_cache

- Assessment: measured; inspect warnings.
- Source log: `user-provided longer RTX measurement transcript`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "no_cache", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 0; see JSONL/run JSON for individual values. These may include warmup transfers.

### reported-long-graphs-on-gpu_hit

- Assessment: failed during cache preparation; no timed GPU-hit inference samples.
- Source log: `user-provided longer RTX measurement transcript`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `True`.
- Transfer log samples: 25; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: AssertionError: slot_mapping.size=0, token_mask.sum()=2048
- Warning: AttributeError: 'NoneType' object has no attribute 'suppress'
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### no_cache-20261110T085850Z-RUtt2Q

- Assessment: measured; inspect warnings.
- Source log: `user-provided unprofiled round1 transcript`; exit code: `0` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "layer_granularity": "-1", "layerwise": "0", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "no_cache", "skip_prefetch": "False", "timed_iters": "50", "warmup_iters": "50"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 0; see JSONL/run JSON for individual values. These may include warmup transfers.

### gpu_hit-20261110T085900Z-bBzka0

- Assessment: failed during preparation without profiling; no timed GPU-hit samples.
- Source log: `user-provided unprofiled round1 transcript`; exit code: `1` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "False", "history_len": "1024", "layer_granularity": "-1", "layerwise": "0", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False", "timed_iters": "50", "warmup_iters": "50"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 25; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: AssertionError: slot_mapping.size=0, token_mask.sum()=2048
- Warning: AttributeError: 'NoneType' object has no attribute 'suppress'
- Warning: CUDA IPC cleanup warning
- Warning: FlexKV destructor error after shutdown

### gpu_hit-20261110T090542Z-unFSqH

- Assessment: failed during preparation on allocation 19; corrupted page offsets persist after synchronization; zero timed samples.
- Source log: `user-provided graph-disabled allocator diagnostic transcript`; exit code: `1` (null means imported transcript, exit status not captured).
- Config: `{"append_history_len": "0", "batch_size": "1", "disable_cudagraph": "True", "history_len": "1024", "layer_granularity": "-1", "layerwise": "0", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False", "timed_iters": "50", "warmup_iters": "50"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 18; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: RuntimeError: Empty GPU page mapping for a nonempty history before FlexKV PUT; see [CacheMetadataError]. No retry or cache fallback was applied.
- Cache metadata error: `{"allocation_request": 19, "gpu_cached_lengths": [0], "history_lengths": [2048], "host_cached_lengths": [0], "page_indices_numel": 64, "page_indptr_after_sync": [1007943680, 0], "page_indptr_before_sync": [1007943680, 0], "rebuilt_slot_mapping_lengths": [0], "slot_mapping_lengths": [0], "user_ids": [28]}`.

### gpu_hit-20261110T093804Z-eX4F8j

- Assessment: diagnostic allocation synchronization; measured; inspect warnings.
- Source log: `/root/recsys/junyiq/recsys-flexkv-minimal/results-flexkv-alloc-sync-check/runs/gpu_hit-20261110T093804Z-eX4F8j/gpu_hit.log`; exit code: `0` (null means imported transcript, exit status not captured).
- Config: `{"allocator_impl": "unmarked", "append_history_len": "0", "batch_size": "1", "disable_cudagraph": "True", "history_len": "1024", "layer_granularity": "-1", "layerwise": "0", "mode": "only_onboard", "num_candidates": "256", "prefetch_gap_ms": "0.0-20.0", "scenarios": "gpu_hit", "skip_prefetch": "False", "sync_cache_alloc": "True", "timed_iters": "50", "warmup_iters": "50"}`.
- Image: `hstu-flexkv-minimal:sm120`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 50; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: CUDA IPC cleanup warning
- Warning: Diagnostic CPU wait before cache allocation changes scheduling and latency

### allocator-validation-1SfHJj

- Assessment: native image build and events_v1 import check passed; regression tests and benchmark did not start; no latency measured.
- Source log: `/root/recsys/junyiq/recsys-flexkv-minimal/results-flexkv-allocfix/validation-1SfHJj/build.log and regression.log (user-provided transcript)`; exit code: `None` (null means imported transcript, exit status not captured).
- Config: `{"benchmark_started": "False", "phase": "build_and_regression"}`.
- Image: `hstu-flexkv-minimal:sm120-allocfix`; GPU selection: `device=2`; profiled: `False`.
- Transfer log samples: 0; see JSONL/run JSON for individual values. These may include warmup transfers.
- Warning: Regression launcher: /usr/bin/python: No module named pytest
