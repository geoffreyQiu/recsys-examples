# HSTU AOTI Inference with KV Cache

## Purpose

This document describes how to export and run the HSTU ranking inference model
with PyTorch AOTInductor (AOTI), native C++ replay, and a FlexKV-backed KV cache.

The supported workflow covers four stages:

1. Building the required custom operators and runtime libraries
2. Exporting the HSTU ranking model with `torch.export` and AOTInductor
3. Starting the FlexKV-backed KV-cache runtime service
4. Validating the exported artifacts and replay tensors with native C++

For lower-level build and deployment details, see the
[HSTU KV-cache AOTI setup guide](./guide_to_hstu_aoti_inference_setup.md).

---

## Scope

> [!NOTE]
> The NVE 26.06/26.07 upgrade covers DynamicEmb and HSTU Python export,
> NVE-aware Python AOTI reload, and native C++ replay. Triton deployment is not
> validated or supported in this stage, and `nve_init_hook/` is not built.
> The existing Triton files remain checked in for a later integration update.

This guide covers these checked-in entry points:

PyTorch export and AOTI validation:

- `export_inference_gr_ranking.py`
- `export_inference_gr_ranking_kvcache.py`

C++ AOTI replay:

- `cpp_inference/inference_hstu_gr_ranking_exported_model.cpp`
- `cpp_inference/inference_hstu_gr_ranking_kvcache_exported_model.cpp`

FlexKV server launcher:

- `start_flexkv_server_for_kvcache_cpp.py`

For each export variant, Python validation and native C++ replay consume the
same model package and replay tensors.

The complete example below focuses on the KV-cache variant. The non-KV-cache
exporter and C++ replay executable remain available for direct AOTI validation.

At the end of the KV-cache workflow, the following key artifacts are expected:

1. Exported KV-cache AOTI package in `examples/hstu/inference_aoti/hstu_gr_ranking_kvcache_model/`
2. Replay tensors in `examples/hstu/inference_aoti/export_test_dump/`
3. C++ executable at `examples/hstu/inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
4. AOTI runtime libraries under `examples/hstu/triton_libs/`

---

## Exported Model Package

The AOTI export path converts the HSTU PyTorch model with `torch.export` and
`torch._inductor.aoti_compile_and_package` for native C++ loading.

The embedding implementation combines DynamicEmb `InferenceEmbeddingTable`
and `ScoredHashTable` with NVEmbedding layers. NVE's `export_aot` stores
schema-v2 layer/resource metadata and embedding-table
files alongside the model `.pt2` archive. Python validation uses NVE's
`load_aot`, and native replay constructs `LayerDirectory` after the AOTI package
loader so each compiled marker constant is rebound to its loaded NVE layer.
The complete exported model archive has this structure:

```text
path/to/model_archive
        ├── model.pt2                              # AOTI model package
        ├── metadata.json                          # NVE schema-v2 metadata
        └── weights/{resource_id}.nve              # NVE storage-resource data
```

---

## Environment

1. PyTorch export, C++ replay, and development use the image built from
   `docker/Dockerfile`. It extends NVIDIA PyTorch 26.05
   (`nvcr.io/nvidia/pytorch:26.05-py3`) with the repository's FBGEMM, FlexKV,
   NVE 26.07, HSTU, DynamicEmb, commons, and KV-cache manager builds. The
   integration uses APIs shared by NVE 26.06 and 26.07. See the HSTU example
   [README](../README.md) for broader training and inference context.

   The image selects NVE's non-plugin key/cache kernel features because this
   workflow uses LinearUVM layers and does not require the optional parameter
   server storage plugins.

2. KV-cache AOTI support depends on the FlexKV source under
   `third_party/FlexKV`, which `docker/Dockerfile` copies into the image.

---

## Container Paths

The commands below assume these paths inside the containers:

| Purpose | Container path |
| --- | --- |
| Repository | `/workspace/recsys-examples` |
| HSTU example | `/workspace/recsys-examples/examples/hstu` |
| Gin configuration | `/workspace/recsys-examples/examples/hstu/inference/configs/kuairand_1k_inference_ranking.gin` |
| Checkpoint | `/workspace/recsys-examples/examples/hstu/ckpt/kuairand_1k_ckpt` |
| KV-cache configuration | `/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml` |

If your layout differs, update the corresponding volume mounts and commands.

---

## Example: HSTU AOTI Inference on KuaiRand-1K

The following commands build the images, prepare KuaiRand-1K data, train a
small checkpoint, export a KV-cache AOTI package, and validate it with native
C++.

### 1. Build the development image

Use a completed `base_triton` image as `BASE_TRITON_IMAGE`. This makes the
Docker build begin at the `devel` stage, skipping the Dockerfile's
`base_fbgemm` and `base_triton` stages while building the remaining
dependencies, in-tree custom operators, C++ replay executable, and runtime
libraries. It intentionally does not build the legacy NVE Triton
initialization hook.

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --build-arg BASE_TRITON_IMAGE="${BASE_TRITON_IMAGE}" \
  -t "recsys-examples-dev" \
  -f "docker/Dockerfile" .
```

Without the `BASE_TRITON_IMAGE` override, the default `base_triton` value
selects the Dockerfile's internal stage and therefore also builds its
`base_fbgemm` dependency.

For fast upgrade validation when CI already published a completed main-branch
`build` image, use `docker/Dockerfile.nve_overlay`. This path replaces and
rebuilds only NVE, updates the changed DynamicEmb/HSTU Python sources, and
incrementally rebuilds the two NVE-aware replay executables. It verifies that
the installed FBGEMM HSTU binary is unchanged. The base must be the final image
from `docker/Dockerfile`, not `base_fbgemm` or `base_triton`:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --platform linux/amd64 \
  --build-arg BASE_MAIN_IMAGE="${BASE_MAIN_IMAGE}" \
  -t "recsys-examples:nve-26.07-overlay" \
  -f "docker/Dockerfile.nve_overlay" .
```

### 2. Prepare the dataset and train a checkpoint

This step preprocesses the `kuairand-1k` dataset, runs single-GPU training with
`./training/configs/kuairand_1k_ranking.gin`, and saves the final checkpoint in
the `model_ckpt` volume for step 3.

Training is supported on Ampere, Hopper, and Blackwell SM100 GPUs.

```bash
docker volume create recsys-data
docker volume create model_ckpt

docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --hostname $(hostname) --name recsys-dev-training \
  --tmpfs /tmp:exec \
  recsys-examples-dev \
  bash -lecx "
    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/
    export CUDA_VISIBLE_DEVICES=0

    cd /workspace/recsys-examples/examples/commons
    python3 ./hstu_data_preprocessor.py \
      --dataset_name kuairand-1k \
      --dataset_path /workspace/recsys-examples/examples/hstu/tmp_data
  "

docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --hostname $(hostname) --name recsys-dev-training \
  --tmpfs /tmp:exec \
  recsys-examples-dev \
  bash -lecx "
    cd /workspace/recsys-examples/examples/hstu
    cp ./training/configs/kuairand_1k_ranking.gin /tmp/kuairand_1k_ranking_train_200.gin
    printf '\nTrainerArgs.log_interval = 50\nTrainerArgs.max_train_iters = 200\nTrainerArgs.ckpt_save_interval = 200\nTrainerArgs.ckpt_save_dir = \"./ckpt\"\n' >> /tmp/kuairand_1k_ranking_train_200.gin

    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
      ./training/pretrain_gr_ranking.py \
      --gin-config-file /tmp/kuairand_1k_ranking_train_200.gin
    rm -rf ./ckpt/kuairand_1k_ckpt
    cp -apr ./ckpt/iter200 ./ckpt/kuairand_1k_ckpt
  "
```

### 3. Export and validate the KV-cache AOTI model

AOTI export and inference are supported on Ampere, Ada, and Blackwell SM120
GPUs.

```bash
docker volume create exported_hstu_model
docker run \
  --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus 1 \
  --volume recsys-data:/workspace/recsys-examples/examples/hstu/tmp_data \
  --volume model_ckpt:/workspace/recsys-examples/examples/hstu/ckpt \
  --volume exported_hstu_model:/exported_hstu_model \
  --hostname $(hostname) --name recsys-dev-inference \
  --tmpfs /tmp:exec \
  recsys-examples-dev \
  bash -lecx "
    export FLEXKV_LOG_LEVEL=WARNING
    export DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/
    export PYTHONPATH=\${PYTHONPATH}:/workspace/recsys-examples/examples/

    cd /workspace/recsys-examples/examples/hstu
    export KVCACHE_MANAGER_CONFIG_FILE=./inference_aoti/kvcache_cpp_runtime.yaml
    python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
      --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
      --checkpoint_dir ./ckpt/kuairand_1k_ckpt \
      --max_bs 2 --kvcache_config_file \${KVCACHE_MANAGER_CONFIG_FILE}

    python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
      --config_file \${KVCACHE_MANAGER_CONFIG_FILE} > flexkv_cache_server.log 2>&1 &
    kvserver_pid=\$!
    sleep 10
    kill -0 \${kvserver_pid}
    ./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
      ./inference_aoti/hstu_gr_ranking_kvcache_model \
      ./inference_aoti/export_test_dump
    kill \${kvserver_pid} || true

    mkdir -p /exported_hstu_model
    cp -apr /workspace/recsys-examples/examples/hstu/inference_aoti/hstu_gr_ranking_kvcache_model /exported_hstu_model/
    cp -apr /workspace/recsys-examples/examples/hstu/inference_aoti/export_test_dump /exported_hstu_model/ "
```

## Triton status

`docker/Dockerfile.tritonserver`, `nve_init_hook/`, and `triton_aoti/` retain the
pre-upgrade integration for follow-up work. Do not use those paths with NVE
26.06 or 26.07 until their loader contract is migrated and validated.
