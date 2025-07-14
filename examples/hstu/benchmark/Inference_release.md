# HSTU Inference Introduction
This is a brief overview of the HSTU inference with KV cache.

## Installation


1. Build TensorRT-LLM (with HSTU KV cache extension):

The HSTU inference utilize customized KV cache manager from TensorRT-LLM.
The current version is based on the HSTU specialized implementation based on TensorRT-LLM v0.19.0.
```bash
~$ cd ${WORKING_DIR}
~$ git clone -b hstu-kvcache-recsys-examples https://github.com/geoffreyQiu/TensorRT-LLM.git tensorrt-llm-kvcache && cd tensorrt-llm-kvcache
~$ git submodule update --init --recursive
~$ make -C docker release_build CUDA_ARCHS="80-real;86-real"
# This will build a docker image with TensorRT-LLM installed.
```

2. Install the dependencies for Recsys-Examples.

Turn on option `INFERENCEBUILD=1` to skip Megatron installation, which is not required for inference.
```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ TRTLLM_KVCACHE_IMAGE="tensorrt_llm/release:latest" docker build \
    --build-arg BASE_IMAGE=${TRTLLM_KVCACHE_IMAGE} \
    --build-arg INFERENCEBUILD=1 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
``` 

3. Try out the showcase or benchmark.

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ python3 ./benchmark/inference_showcase.py
~$ python3 ./benchmark/paged_hstu_with_kvcache_benchmark.py
``` 

## Key Features

1. Cache for KV data

We use GPU memory and host storage for KV data cache., as in `GpuKVCacheManager` and `HostKVStorageManager`. This can help to reduce the recomputation of KV data.

The GPU KV cache is organized as a paged KV-data table, and supports KV data adding/appending, lookup and eviction. When appending new data to the GPU cache, we will evict data from the oldest users according to the LRU policy if there is no empty page. The HSTU attention kernel also accepts KV data from a paged table.

The host KV data storage support adding/appending and lookup. We only present an example implementation, since this can be built over other database and can vary widely in the deployment.

2. Optimization with CUDA graph

We utilize the graph capture and replay support in Torch for convenient CUDA graph optimization on the HSTU layers. This decreases the overhead for kernel launch, especially for input with a small batch size.

We break down the CUDA graph of HSTU blocks by layers, so that host KV data transfer to GPU can be overlapped with HSTU computation. Also, the input data (hidden states) fed to HSTU layers needs paddding to pre-determined batch size and sequence length, due to the requirement of static shape in CUDA graph.

## Benchmark Results

Here we present the benchmark results of the HSTU layers with KV cache on L20 and L40 gpus.

HSTU Setup for benchmark:

| Parameter | Value |
|-----------|-------|
| Number of HSTU layers | 8 |
| Hidden Dim Size | 1024 |
| Number of Heads | 4 |
| Head Dim Size | 256 |
| Max Batchsize| 16 |
| Max Per Sequence Length | 4096 |
| Per Sequence Targets Number | 256 |


Performance results for HSTU block (8-layers) on L20 GPU:

![Local Image](hstu_inference_l20_batch1.png)
![Local Image](hstu_inference_l20_batch8.png)



