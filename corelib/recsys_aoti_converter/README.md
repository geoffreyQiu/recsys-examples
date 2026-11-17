# History-KV conversion for PyTorch AOTInductor

Convert supported PyTorch attention graphs into one end-to-end
`torch.export.ExportedProgram` and one AOTInductor package with history K/V
cache operations. This tool analyzes the exported ATen FX graph; it does not
select transformations by model class or architecture name.

The runtime uses the shared [corelib cache owner](../recsys_kvcache_manager/README.md).
It keeps fixed-slot metadata and graph adapters in this directory, while the
compiled core library owns transfer dispatch, progress and completion. The
[HSTU AOTI example](../../examples/hstu/inference_aoti/README.md) uses that same
owner through the standard paged-operator interface.

## Build

Dependencies: CUDA-enabled PyTorch with `torch.export` and AOTInductor, FlexKV's
server/client API, NumPy, PyYAML, NVTX, ZeroMQ, CMake, Ninja and a C++/CUDA compiler.
The tool uses private PyTorch APIs. The supplied [Dockerfile](docker/Dockerfile)
pins a public NVIDIA PyTorch image and a public FlexKV revision; a different
PyTorch/FlexKV version requires validation and rebuilding.

From the repository root:

```bash
docker build -f corelib/recsys_aoti_converter/docker/Dockerfile \
  --build-arg CUDA_ARCH=9.0 -t recsys-aoti-converter:dev .
export NV_GPU=GPU-your-assigned-uuid
docker run --rm -it --init --gpus "device=$NV_GPU" --ipc=host \
  -e NV_GPU="$NV_GPU" \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/recsys-examples" recsys-aoti-converter:dev bash
```

Select the architecture for the deployment GPU; `9.0` targets H100 and `12.0`
targets SM120. If the repository filesystem cannot be bind-mounted, stage a
copy on a filesystem Docker can mount, such as `/tmp`.

Inside the prepared environment:

```bash
cd /workspace/recsys-examples/corelib/recsys_aoti_converter
python3 scripts/check_environment.py
python3 scripts/build_native.py --source ../.. --output build
python3 scripts/build_single.py --reference ../..
```

No converter package installation is needed. The first build creates the shared
manager and append library; the second links the converter adapter to that
manager. Keep `build/manager`, `build/append` and `build/single` together. Use the
matching sources in this branch, including the FP32 append support and
mutation-aware append overload. An older manager without `CacheOwner` is
incompatible. AOTI packages can contain CPU-specific instructions as well as GPU
kernels: compile for a CPU/GPU architecture compatible with deployment.

## Synthetic example

The example generates random weights and inputs locally. No checkpoint, dataset
or generated model graph is distributed with the tool. Run from this directory:

```bash
python3 examples/prepare.py --output /tmp/history_example
python3 scripts/convert_single.py \
  --package /tmp/history_example \
  --model-spec /tmp/history_example/model.json \
  --spec /tmp/history_example/cache.json --reference ../.. \
  --precision bf16 --no-autocast --sdpa --review-diff \
  --max-batch 8 --output artifacts/toy_bf16
python3 scripts/run_sample.py --bundle artifacts/toy_bf16 \
  --reference ../.. --batch-size 4 --cpu-roundtrip
python3 scripts/audit_single.py --bundle artifacts/toy_bf16
```

Use new package/output directories for subsequent examples. The
[model source](examples/toy_attention.py) embeds immutable history IDs, projects
history K/V, and attends to them using changing query inputs. Inspect generated
`model.json` and `cache.json` to adapt the example to another model.

## Describe a model and its immutable inputs

[CacheSpec](history_cache/convert.py) declares which flattened input tensors
remain identical for a reusable snapshot. For the synthetic example:

```json
{
  "history_input_indices": [0],
  "length": 16,
  "heads": 2,
  "head_dim": 64,
  "policy": "immutable_snapshot"
}
```

`history_input_indices` indexes original flattened input arguments, before cache
inputs are added. It does not index token positions. Dictionary input order is
defined by the descriptor's `input_names` symbol. In the example, index `0` is
`history_ids`; `queries` is not declared immutable.

A model descriptor specifies a Python factory, input order and sample source:

```json
{
  "source": "source",
  "class": "toy_attention:ToyAttention",
  "input_names": "toy_attention:INPUT_NAMES",
  "hyperparameters": "hyperparameters.json",
  "checkpoint": "weights.pt",
  "sample": {"format": "npz", "path": "sample.npz"}
}
```

The source directory is relative to `--package`; hyperparameters and checkpoint
paths are relative to that source directory. The loader calls the factory with
JSON keyword arguments, loads its state dictionary, and constructs a dictionary
of tensors using the declared input order. Samples can use `npz` or saved
`triton_json` request format; loading a sample does not contact a server.

For an existing serialized ATen graph, `format: "export_json"` accepts separate
`graph`, `checkpoint`, `schema` and `sample` paths relative to the package. The
schema's `inputs` array supplies ordered names. This path uses the pinned
PyTorch serializer and validates the saved signature.

## Conversion and execution

```text
original model → torch.export → structural history/attention analysis
                              → one rewritten ExportedProgram → AOTI PT2

lookup / allocation / onboard launch
  → pack missing histories → compute missing K/V → write GPU cache tables
  → wait for CPU restoration → gather K/V → attention
  → optional offload submission → request completion
```

- **Projection matching:** identify history-dependent linear K/V producers,
  head layouts and attention consumers. The projection path currently expects
  `linear → reshape/view → permute(0, 2, 1, 3)`. Unknown layouts and dependency
  patterns are rejected.
- **Rolling windows:** match static slice/concatenation updates; cache the
  immutable prefix for each position transform and compute changing tails in
  request-local storage. Shared cache slots are not appended to by these tails.
- **Layered stacks:** support repeated fused-UVQK SiLU attention when static mask
  analysis proves history cannot attend to changing candidate tokens. Configure
  `history_start` for the history suffix. Each layer has its own cached K/V pair.
- **Attention fusion:** `--sdpa` recognizes supported matmul/softmax attention,
  preserving additive mask values. PyTorch selects the SDPA backend. SiLU
  attention keeps its original arithmetic.
- **Review output:** `--review-diff` emits per-bank annotated diffs, full before/
  after FX code and node-origin maps. It does not reconstruct original Python
  loops or class structure.

| Control | Behavior |
|---|---|
| `--precision bf16` | BF16 floating dataflow, no autocast by default; explicit FP64 state/casts and promoted FP64 values remain FP64. |
| `--autocast` | Optional BF16 mixed precision; accepted only with BF16. |
| `--precision fp32 --no-tf32` / `--tf32` | FP32 export with TF32 disabled/enabled. |
| `--sdpa` / `--no-sdpa` | Enable/disable supported softmax attention fusion; default off. |
| `--review-diff` | Generate readable transformation artifacts. |
| `--max-batch` / `--capacity` | Dynamic batch upper bound (default 8; 2..256) / history-slot capacity (at least max batch). |
| `--owner-loop` / `--no-owner-loop` | Shared owner (default) / legacy direct runtime; does not change the PT2 graph ABI. |
| `--export-only` / `--capture-only` | Stop before AOTI compilation / after original capture and analysis. |
| `--reuse-capture` | Reuse a capture with matching recorded precision and math policy. |

Outputs include `model_export.pt2`, `model_aoti.pt2`, uncached controls, a manifest,
input-tree metadata, dtype audits and conversion diagnostics. With SDPA enabled,
additional uncached SDPA controls are produced. `review/` is optional.

Load a converted bundle through `history_cache.single.SingleRunner`, with a
`history_cache.native.NativeCache` whose geometry, precision, capacity and
maximum batch match the manifest. Applications supply `snapshot_ids: int64[B]`;
an ID may represent `(user_id, history_version)`. Reuse an ID only for identical
declared history and required mask/normalization metadata. Duplicate IDs may
share a slot while queries and request-local tails differ. The wrapper consumes
the private completion output and preserves the original public output tree.

`SingleRunner` validates shape/dtype/layout and snapshot identity. Disable
`verify_history` only when the application enforces the same immutable-ID
contract. `offload=True` requests CPU publication; `cache.flush()` completes it
before explicit GPU eviction. Model reads still wait for required onboarding.
The pinned FlexKV backend retains child task records, so this adapter selects
`onboard_child_tasks: 1` and waits for terminal completion before reaping them.

Complete GPU-resident hits use compact metadata. Optional
`history_cache.cuda_graph.CachedCudaGraphs` captures inference and can overlap
CPU onboarding with independent work by replaying prefix/suffix graph fragments.
Native AOTI execution does not automatically capture CUDA graphs. Runtime
switches `HISTORY_RESIDENT_FASTPATH=0` and `HISTORY_ONBOARD_OVERLAP=0` disable the
corresponding optimizations.

## Limits

- Whole-tensor snapshot declarations; immutable subregions of a changing raw
  input tensor are not a supported contract.
- Fixed/max history length, with power-of-two native slot length. No persistent
  incremental append or history update under an existing snapshot ID.
- One serialized runtime per process; concurrent shared instances are unsupported.
- Contiguous CUDA inputs, fixed non-batch dimensions, batch size within the
  manifest limit. Current append kernels support head dimensions 64/128/256/512.
- Structural checks reject unknown token mixing, batch mixing and unsafe history
  dependencies. Layered reuse also requires valid mask/normalization dependencies.
- Recompile for the precision, PyTorch build and target CPU/GPU architecture.
  BF16 and fused attention can change numerical results and discrete decisions.

## Tests and timing helper

Tests use generated tensors and synthetic models:

```bash
python3 tests/single_precision.py
python3 tests/context_guard_elision.py
python3 tests/test_attention_layout.py
python3 tests/test_append_fill.py
python3 tests/single_native.py --reference ../..
python3 tests/single_conversion.py --reference ../.. \
  --precision fp32 --output artifacts/test_fp32
python3 tests/single_conversion.py --reference ../.. \
  --precision bf16 --output artifacts/test_bf16
python3 tests/layered_history.py --reference ../.. \
  --precision bf16 --output artifacts/test_layered
```

To time a converted bundle without recompilation:

```bash
python3 scripts/benchmark_aoti.py --bundle artifacts/toy_bf16 \
  --reference ../.. --batches 1 2 4 8 --warmup 10 --repeats 50
python3 scripts/format_perf.py --input artifacts/toy_bf16/performance.json \
  --output artifacts/toy_bf16/performance.md --title "Model latency"
```

The helper measures sequential, CUDA-synchronized forward latency. Cache lookup,
onboarding and waits are included; compilation, capture, CPU publication,
eviction and HTTP/Triton transport are excluded. Inputs come from the captured
example and repeat when necessary. Runtime flags, route statistics and raw
samples are saved alongside the matrix. No benchmark results are distributed.
