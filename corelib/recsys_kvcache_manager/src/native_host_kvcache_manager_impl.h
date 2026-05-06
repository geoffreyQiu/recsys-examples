/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <barrier>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "nvcomp/ans.h"

namespace kvcache {


class HostKVStorageImpl;

class KVOnloadHandle {
public:
    KVOnloadHandle(int num_layers);
    ~KVOnloadHandle();

    void init();
    void reset();
    void complete_host(int layer_idx, cudaStream_t stream);
    void wait_host(int layer_idx);

public:
    int num_layers;
    std::vector<cudaEvent_t> compl_event;
    std::vector<cudaEvent_t> internal_onload_event;
    std::vector<int> host_complete;
    std::mutex mtx_;
    std::condition_variable cv_;

    bool inited;
    bool no_onload;
};

class KVOffloadHandle {
public:
    KVOffloadHandle(int num_layers);
    ~KVOffloadHandle();

    void init();
    void reset();
    void complete_host(int layer_idx, cudaStream_t stream, 
                       std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >>&& chunks);
    bool try_wait_host(int layer_idx);
    float get_launch_time(void) { return this->time_stamp; }
    void set_launch_time(float time) { this->time_stamp = time; }

public:
    int num_layers;
    std::vector<cudaEvent_t> ready_event;
    std::vector<cudaEvent_t> internal_gather_event;
    std::vector<std::atomic<int>> host_ready;

    std::vector<std::pair<int32_t, int32_t>> pages;
    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> chunks;
    
    bool inited;
    bool no_offload;
    float time_stamp;
};


class HostKVStorageImpl
{
public:
    HostKVStorageImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int64_t num_tokens_per_chunk
    );
    ~HostKVStorageImpl();

    int64_t get_kvdata_length(int64_t user_id);

    void append_kvdata(
        int64_t user_id, int64_t start_position, int64_t length, 
        uint16_t *kvdata_buffer, size_t buffer_layer_stride);
    void append_kvdata(
        int64_t user_id, int64_t start_position, int64_t length, 
        uint16_t *kvdata_buffer, size_t buffer_layer_stride,
        size_t *kvdata_bytes, size_t bytes_layer_stride);

    std::vector<uint16_t*> get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx);
    std::vector<size_t> get_kvdata_bytes(int64_t user_id, int64_t length, int64_t layer_idx);

public:
    std::vector<at::Tensor> get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat = true);
    void init_random_kvdata(int64_t user_id, size_t num_tokens);

public:
    const int num_layers;
    const int num_kv_heads;
    const int kv_headdim;
    const int page_size;

    const int64_t chunk_size;
    size_t chunk_numel;
    size_t page_numel;
    size_t per_token_numel;
    size_t layer_numel;

    // all chunk sizes are measured as pages
    std::queue<std::pair<int64_t, int64_t>> _empty_chunks;  // <offset of buffer, size in pages>
    int64_t _empty_sizes;
    // std::list<int64_t> _lru_list;
    // std::unordered_map<int64_t, 
    //                 typename std::list<int64_t>::iterator> _lru_lookup_table;

    std::unordered_map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> _uid_to_chunks;
    std::unordered_map<int64_t, int64_t> _uid_to_length;

    std::mutex host_kvcache_mutex_;

public:
    std::vector<void*> onload_gpu_buffers;
    std::vector<void*> offload_gpu_buffers;
    std::vector<void*> pinned_kvstorage_buffers;
};


}  // namespace kvcache