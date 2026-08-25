# HSTU AOTI Dual-NVE Build and Loader Guide

This guide describes how the development image builds and selects NVE 26.05
and 26.07. The [AOTI README](./README.md) contains the complete copy-and-paste
export, native replay, and Triton command sequences.

## Installed layout

The development image builds both pinned NVE source trees independently. NVE
26.07 is copied from the parent repository's pinned
`third_party/nv-embedding-cache` submodule, while NVE 26.05 is cloned at its
pinned commit inside the Docker build:

```text
/workspace/deps/nve-26.05
/workspace/deps/nve-26.07
/workspace/deps/NVTX
/workspace/deps/nlohmann-json
```

The Python packages and native libraries are installed under separate roots:

```text
/opt/nve/26.05/python/pynve
/opt/nve/26.07/python/pynve
```

Both builds use NVE's upstream default features, including
`libnve-plugin-nvhm.so`, so normal hierarchical HSTU inference can construct
its default `NVEParameterServer` backend. There is no unversioned `pynve`
package in the development image and neither NVE library directory appears in
the global `LD_LIBRARY_PATH` or `LD_PRELOAD`.

The image default is:

```text
PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu
```

Selecting 26.05 replaces that value for a fresh Python process:

```bash
PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu python3 ...
```

The two `/workspace/recsys-examples` entries are code roots, not NVE
installations. Version selection still comes from exactly one `/opt/nve` root.

Appending `/opt/nve/26.05/python` to the image's inherited `PYTHONPATH` is not
supported because it leaves both generations visible and makes import order
the selector.

## Python constructor and export compatibility

The source maps the two constructor APIs as follows:

| Logical layer | NVE 26.05 | NVE 26.07 |
| --- | --- | --- |
| GPU-only | `cache_type=CacheType.NoCache` | `layer_type=LayerType.GPULayer` |
| Linear UVM | `cache_type=CacheType.LinearUVM` | `layer_type=LayerType.LinearUVM` |
| Hierarchical | `cache_type=CacheType.Hierarchical`, `remote_interface=...` | `layer_type=LayerType.Hierarchical`, `storage=...` |

NVE 26.05 and 26.07 also expose incompatible
`nve_ops::embedding_lookup` schemas. HSTU imports NVE first and installs its
dynamic-size two-argument fake with `allow_override=True` only under 26.05.
Under 26.07, HSTU performs no duplicate registration and retains NVE's native
four-argument fake.

Both exporters call the selected package's `export_aot`. Reload is routed
through `nve_aoti_compat.py`:

1. Parse `<package>/metadata.json`.
2. Classify only an unambiguous 26.05 legacy shape or 26.07 schema-v2 shape.
3. Compare it with the imported `pynve` generation.
4. For 26.05, recreate/register legacy layers and then construct the ordinary
   AOTI package loader.
5. For 26.07, call NVE's marker-aware `load_aot`.
6. Keep loaded layers alive and normalize the runner result to a tensor list.
7. Release the AOTI loader before releasing its NVE layers.

## Native replay build

The Docker build configures the native targets with exact versioned roots:

```bash
set -euo pipefail
cd /workspace/recsys-examples

export CMAKE_PREFIX_PATH="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')"

cmake -S ./examples/hstu/inference_aoti/cpp_inference \
  -B ./examples/hstu/inference_aoti/cpp_inference/build \
  -DNVE_2605_ROOT=/workspace/deps/nve-26.05 \
  -DNVE_2607_ROOT=/workspace/deps/nve-26.07 \
  -DNVE_2605_LIB_DIR=/opt/nve/26.05/python/pynve \
  -DNVE_2607_LIB_DIR=/opt/nve/26.07/python/pynve \
  -DNVTX_INCLUDE_DIR=/workspace/deps/NVTX/c/include \
  -DNLOHMANN_JSON_ROOT=/workspace/deps/nlohmann-json

cmake --build ./examples/hstu/inference_aoti/cpp_inference/build -j 8
cmake --install ./examples/hstu/inference_aoti/cpp_inference/build
```

This produces two NVE-independent workflow executables:

```text
examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_exported_model
examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model
```

It also compiles one CUDA adapter source twice and installs two plugins:

```text
/opt/nve/26.05/replay/librecsys_nve_loader.so
/opt/nve/26.07/replay/librecsys_nve_loader.so
```

The executable and its main-side manager include no NVE headers and have no
NVE `DT_NEEDED` entry. The manager accepts only `NVE_VERSION=26.05` or
`NVE_VERSION=26.07`; an environment value cannot name an arbitrary DSO. Each
plugin has a relative RUNPATH to its matching `../python/pynve` directory.

## Native lifecycle

Native-op registration and generation-specific layer-state construction are
separate phases:

| Step | NVE 26.05 | NVE 26.07 |
| --- | --- | --- |
| Validate | Check `NVE_VERSION` and legacy metadata | Check `NVE_VERSION` and schema-v2 metadata |
| Load | `dlopen` 26.05 plugin and prepare native ops | `dlopen` 26.07 plugin and prepare native ops |
| Before AOTI | Construct legacy `LayerDirectory(package, device)` | No layer state yet |
| AOTI | Construct `AOTIModelPackageLoader` | Construct `AOTIModelPackageLoader` after native ops are registered |
| After AOTI | Replay | Construct marker-aware `LayerDirectory(package, loader, device, resources)`, then replay |
| Shutdown | Destroy AOTI loader, then layer state | Destroy AOTI loader, then marker/resource state |

The selected plugin and its registered operator libraries remain resident
until process termination. In-process version switching and `dlclose` after
operator registration are unsupported.

## Exact scenario matrix

The four exporter/replayer pairs use these fixed selectors and paths. Each
exporter command also requires the common Gin/checkpoint arguments shown in
the [README export section](./README.md#export-all-four-python-scenarios).

| Scenario | Export selector and output arguments | Replay selector and positional arguments |
| --- | --- | --- |
| 26.05 non-KV | `PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu`, `--export_dir /exported_hstu_model/nve-26.05/non-kv/model`, `--dump_dir /exported_hstu_model/nve-26.05/non-kv/replay` | `NVE_VERSION=26.05`, package `/exported_hstu_model/nve-26.05/non-kv/model`, dump `/exported_hstu_model/nve-26.05/non-kv/replay` |
| 26.07 non-KV | `PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu`, `--export_dir /exported_hstu_model/nve-26.07/non-kv/model`, `--dump_dir /exported_hstu_model/nve-26.07/non-kv/replay` | `NVE_VERSION=26.07`, package `/exported_hstu_model/nve-26.07/non-kv/model`, dump `/exported_hstu_model/nve-26.07/non-kv/replay` |
| 26.05 KV-cache | `PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu`, `--export_dir /exported_hstu_model/nve-26.05/kv-cache/model`, `--dump_dir /exported_hstu_model/nve-26.05/kv-cache/replay` | `NVE_VERSION=26.05`, package `/exported_hstu_model/nve-26.05/kv-cache/model`, dump `/exported_hstu_model/nve-26.05/kv-cache/replay` |
| 26.07 KV-cache | `PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu`, `--export_dir /exported_hstu_model/nve-26.07/kv-cache/model`, `--dump_dir /exported_hstu_model/nve-26.07/kv-cache/replay` | `NVE_VERSION=26.07`, package `/exported_hstu_model/nve-26.07/kv-cache/model`, dump `/exported_hstu_model/nve-26.07/kv-cache/replay` |

KV-cache replay additionally requires the FlexKV service configured by:

```text
/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml
```

## Triton is fixed to NVE 26.05

The Docker build compiles the unchanged init hook with:

```text
NVE_ROOT=/workspace/deps/nve-26.05
NVE_LIB_DIR=/opt/nve/26.05/python/pynve
```

`docker/Dockerfile.tritonserver` copies that 26.05 source to the existing
`/workspace/deps/nve` runtime location, copies the 26.05 package to global
`site-packages/pynve`, and populates `examples/hstu/triton_libs/pynve` from
26.05 only. It does not copy the 26.07 prefix or submodule source tree.

The two supported Triton demos are:

| Workflow | Artifact | Model | Client selector |
| --- | --- | --- | --- |
| Non-KV | `/exported_hstu_model/nve-26.05/non-kv/model` | `hstu_gr_ranking` | `--workflow non-kv` |
| KV-cache | `/exported_hstu_model/nve-26.05/kv-cache/model` | `hstu_gr_ranking_kvcache` | `--workflow kv-cache` |

Use the complete server and client commands in the
[README Triton section](./README.md#run-the-2605-only-triton-demos). NVE 26.07
cannot use the current hook because marker rebinding needs an AOTI loader that
does not exist when the backend calls the hook.

## Expected failures

The Python compatibility loader and native manager fail with a concise error
for missing, empty, mixed, or unsupported metadata. A cross-generation load
uses this form:

```text
NVE version mismatch: selected runtime 26.07, artifact requires 26.05
```

Native replay also fails before state construction for a missing/unsupported
`NVE_VERSION`, plugin ABI or structure mismatch, wrong claimed generation,
wrong init phase, or incomplete function table.
