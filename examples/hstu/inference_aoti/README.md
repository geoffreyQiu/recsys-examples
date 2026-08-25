# HSTU AOTI Inference with NVE 26.05 and 26.07

This directory contains the non-KV and KV-cache HSTU AOTInductor exporters,
their native C++ replayers, and the 26.05-only Triton demos.

## Supported workflows

| Workflow | NVE 26.05 | NVE 26.07 |
| --- | ---: | ---: |
| Normal HSTU Python inference | Supported with an exact `PYTHONPATH` | Supported and the development-image default |
| Non-KV Python export/reload | Supported | Supported |
| KV-cache Python export/reload | Supported | Supported |
| Non-KV native replay | `NVE_VERSION=26.05` | `NVE_VERSION=26.07` |
| KV-cache native replay | `NVE_VERSION=26.05` | `NVE_VERSION=26.07` |
| Triton AOTI deployment | Supported | Unsupported by the current init-hook contract |

Each process uses exactly one NVE generation. Do not put both versioned Python
prefixes in `PYTHONPATH`, do not append the selected prefix to an inherited
`PYTHONPATH`, and do not change versions after `pynve` has been imported.

Python selection is exact:

```bash
PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu python3 ...
PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu python3 ...
```

The last two entries expose repository Python modules; only the first entry
selects NVE, and exactly one versioned NVE prefix is present in either command.

Native replay uses one NVE-independent executable per workflow. It reads
`NVE_VERSION` once and maps it to one compiled plugin path:

```text
26.05 -> /opt/nve/26.05/replay/librecsys_nve_loader.so
26.07 -> /opt/nve/26.07/replay/librecsys_nve_loader.so
```

The Triton image contains only NVE 26.05 and retains the existing
`LD_LIBRARY_PATH`, `LD_PRELOAD`, and model-init-hook behavior.

## Artifact contract

Both exporters create an isolated directory with this layout:

```text
model/
├── model.pt2
├── metadata.json
└── weights/
```

NVE 26.05 writes legacy array metadata with `cache_type`; NVE 26.07 writes a
schema-v2 object with `version: 2`, `resources`, and `layer_type`. Python reload
and native replay inspect this NVE-owned file and reject a runtime/artifact
mismatch. Artifacts are not portable across NVE generations.

`--export_dir` and `--dump_dir` must be distinct and non-nested. Each may be
absent or empty; the exporter refuses a non-empty destination so a rerun cannot
merge stale model or replay data.

## Build the images

Run from the repository root on the Docker host:

```bash
set -euo pipefail

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --target base_fbgemm \
  -t recsys-fbgemm-base \
  -f docker/Dockerfile .

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --build-arg BASE_FBGEMM_IMAGE=recsys-fbgemm-base \
  -t recsys-examples-dev \
  -f docker/Dockerfile .

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --build-arg PYTORCH_AOTI_IMAGE=recsys-examples-dev \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.06-py3 \
  -t recsys-examples-triton-nve-2605 \
  -f docker/Dockerfile.tritonserver .
```

The development build installs both NVE generations from pinned source,
defaults Python to 26.07, builds both native replay plugins, and builds the
existing Triton init hook against 26.05. The dedicated Triton image copies only
the 26.05 source, package, libraries, and hook inputs.

## Start the development container

The commands below assume the existing data and checkpoint volumes have been
populated and use a separate volume for generated artifacts.

Run on the Docker host:

```bash
set -euo pipefail

docker volume create recsys-data
docker volume create model_ckpt
docker volume create exported_hstu_model

docker run --rm -it \
  --name recsys_dual_nve_dev \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --tmpfs /tmp:exec \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --volume exported_hstu_model:/exported_hstu_model \
  recsys-examples-dev \
  bash
```

## Export all four Python scenarios

Run each scenario in a fresh development-container process or shell whose
`pynve` has not already been imported. The parent directories below must be
new for the first run.

### NVE 26.05, non-KV

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu
install -d /exported_hstu_model/nve-26.05/non-kv

PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu \
DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build \
CUDA_VISIBLE_DEVICES=0 \
python3 ./inference_aoti/export_inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
  --max_bs 2 \
  --export_dir /exported_hstu_model/nve-26.05/non-kv/model \
  --dump_dir /exported_hstu_model/nve-26.05/non-kv/replay
```

### NVE 26.07, non-KV

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu
install -d /exported_hstu_model/nve-26.07/non-kv

PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu \
DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build \
CUDA_VISIBLE_DEVICES=0 \
python3 ./inference_aoti/export_inference_gr_ranking.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
  --max_bs 2 \
  --export_dir /exported_hstu_model/nve-26.07/non-kv/model \
  --dump_dir /exported_hstu_model/nve-26.07/non-kv/replay
```

### NVE 26.05, KV-cache

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu
install -d /exported_hstu_model/nve-26.05/kv-cache

PYTHONPATH=/opt/nve/26.05/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu \
DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build \
CUDA_VISIBLE_DEVICES=0 \
python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
  --max_bs 2 \
  --kvcache_config_file ./inference_aoti/kvcache_cpp_runtime.yaml \
  --export_dir /exported_hstu_model/nve-26.05/kv-cache/model \
  --dump_dir /exported_hstu_model/nve-26.05/kv-cache/replay
```

### NVE 26.07, KV-cache

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu
install -d /exported_hstu_model/nve-26.07/kv-cache

PYTHONPATH=/opt/nve/26.07/python:/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu \
DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build \
CUDA_VISIBLE_DEVICES=0 \
python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
  --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
  --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
  --max_bs 2 \
  --kvcache_config_file ./inference_aoti/kvcache_cpp_runtime.yaml \
  --export_dir /exported_hstu_model/nve-26.07/kv-cache/model \
  --dump_dir /exported_hstu_model/nve-26.07/kv-cache/replay
```

Each exporter immediately reloads its artifact with the same selected runtime
and checks the compiled output before producing replay data.

## Run all four native replay scenarios

### Non-KV, NVE 26.05

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu

NVE_VERSION=26.05 \
./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_exported_model \
  /exported_hstu_model/nve-26.05/non-kv/model \
  /exported_hstu_model/nve-26.05/non-kv/replay
```

### Non-KV, NVE 26.07

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu

NVE_VERSION=26.07 \
./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_exported_model \
  /exported_hstu_model/nve-26.07/non-kv/model \
  /exported_hstu_model/nve-26.07/non-kv/replay
```

### KV-cache, NVE 26.05

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu

export KVCACHE_MANAGER_CONFIG_FILE=/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml
PYTHONPATH=/workspace/recsys-examples/examples \
python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
  --config_file "${KVCACHE_MANAGER_CONFIG_FILE}" \
  >/tmp/hstu-flexkv-nve-2605.log 2>&1 &
flexkv_pid=$!

cleanup_flexkv() {
  kill "${flexkv_pid}" 2>/dev/null || true
  wait "${flexkv_pid}" 2>/dev/null || true
}
trap cleanup_flexkv EXIT

sleep 10
kill -0 "${flexkv_pid}"

NVE_VERSION=26.05 \
./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
  /exported_hstu_model/nve-26.05/kv-cache/model \
  /exported_hstu_model/nve-26.05/kv-cache/replay
```

### KV-cache, NVE 26.07

```bash
set -euo pipefail
cd /workspace/recsys-examples/examples/hstu

export KVCACHE_MANAGER_CONFIG_FILE=/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml
PYTHONPATH=/workspace/recsys-examples/examples \
python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
  --config_file "${KVCACHE_MANAGER_CONFIG_FILE}" \
  >/tmp/hstu-flexkv-nve-2607.log 2>&1 &
flexkv_pid=$!

cleanup_flexkv() {
  kill "${flexkv_pid}" 2>/dev/null || true
  wait "${flexkv_pid}" 2>/dev/null || true
}
trap cleanup_flexkv EXIT

sleep 10
kill -0 "${flexkv_pid}"

NVE_VERSION=26.07 \
./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
  /exported_hstu_model/nve-26.07/kv-cache/model \
  /exported_hstu_model/nve-26.07/kv-cache/replay
```

The executable rejects a missing or unsupported `NVE_VERSION`, invalid
metadata, and a runtime/artifact mismatch before loading either NVE plugin.

## Run the 26.05-only Triton demos

NVE 26.07 is not supported by the current Triton model-init hook because the
hook runs before the backend creates its AOTI loader.

### Non-KV Triton

Run the server on the Docker host in terminal A:

```bash
set -euo pipefail

docker run --rm \
  --name triton_hstu_nonkv_nve_2605 \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=8G \
  --volume exported_hstu_model:/exported_hstu_model \
  recsys-examples-triton-nve-2605 \
  bash -leuc '
    cd /workspace/recsys-examples/examples/hstu
    model_repo=$(mktemp -d /tmp/hstu-triton-nonkv-2605.XXXXXX)
    install -d "${model_repo}/hstu_gr_ranking/1"
    cp ./inference_aoti/triton_aoti/hstu_gr_ranking/config.pbtxt \
      "${model_repo}/hstu_gr_ranking/config.pbtxt"
    cp -a /exported_hstu_model/nve-26.05/non-kv/model/. \
      "${model_repo}/hstu_gr_ranking/1/"
    exec tritonserver --model-repository="${model_repo}"
  '
```

Run the client on the Docker host in terminal B:

```bash
set -euo pipefail

docker exec triton_hstu_nonkv_nve_2605 \
  python3 /workspace/recsys-examples/examples/hstu/inference_aoti/test_tritonserver_aoti_hstu_model.py \
    --workflow non-kv \
    --dump_dir /exported_hstu_model/nve-26.05/non-kv/replay \
    --url localhost:8000 \
    --model_name hstu_gr_ranking \
    --batch_size 2
```

### KV-cache Triton

Run the server and FlexKV service on the Docker host in terminal A:

```bash
set -euo pipefail

docker run --rm \
  --name triton_hstu_kv_nve_2605 \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=8G \
  --volume exported_hstu_model:/exported_hstu_model \
  recsys-examples-triton-nve-2605 \
  bash -leuc '
    cd /workspace/recsys-examples/examples/hstu
    model_repo=$(mktemp -d /tmp/hstu-triton-kv-2605.XXXXXX)
    install -d "${model_repo}/hstu_gr_ranking_kvcache/1"
    cp ./inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt \
      "${model_repo}/hstu_gr_ranking_kvcache/config.pbtxt"
    cp -a /exported_hstu_model/nve-26.05/kv-cache/model/. \
      "${model_repo}/hstu_gr_ranking_kvcache/1/"

    export FLEXKV_LOG_LEVEL=WARNING
    export KVCACHE_MANAGER_CONFIG_FILE=/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml
    python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
      --config_file "${KVCACHE_MANAGER_CONFIG_FILE}" \
      >/tmp/hstu-flexkv-triton-2605.log 2>&1 &
    flexkv_pid=$!
    sleep 10
    kill -0 "${flexkv_pid}"

    exec tritonserver --model-repository="${model_repo}"
  '
```

Run the client on the Docker host in terminal B:

```bash
set -euo pipefail

docker exec triton_hstu_kv_nve_2605 \
  python3 /workspace/recsys-examples/examples/hstu/inference_aoti/test_tritonserver_aoti_hstu_model.py \
    --workflow kv-cache \
    --dump_dir /exported_hstu_model/nve-26.05/kv-cache/replay \
    --url localhost:8000 \
    --model_name hstu_gr_ranking_kvcache \
    --batch_size 2
```

For build-graph details and the loader lifecycle, see
[guide_to_hstu_aoti_inference_setup.md](./guide_to_hstu_aoti_inference_setup.md).
