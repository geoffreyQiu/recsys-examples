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

#include "kvcache_manager_impl.h"
#include <nvtx3/nvtx3.hpp>
#include "nvcomp/ans.h"

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void scatter_paged_kvcache(
    uint16_t *paged_kvcache_table,    // output
    uint16_t *continuous_gpu_buffer,  // input: gpu onload buffer
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream);

void gather_paged_kvcache(
    uint16_t *gather_gpu_buffer,    // output: gpu offload buffer
    uint16_t *paged_kvcache_table,  // input
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream);

namespace kvcache {


KVOnloadHandle::KVOnloadHandle() : no_onload(true) {}

KVOnloadHandle::KVOnloadHandle(
    int num_layers
)
: num_layers(num_layers)
, compl_event(std::vector<cudaEvent_t>(num_layers))
, internal_onload_event(std::vector<cudaEvent_t>(num_layers))
, host_complete(num_layers, 0)
, no_onload(true)
, inited(false) {}

KVOnloadHandle::~KVOnloadHandle(){
    if (!inited) return;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaEventDestroy(compl_event[layer_idx]);
        cudaEventDestroy(internal_onload_event[layer_idx]);
    }
};

void KVOnloadHandle::init() {
    this->no_onload = false;
    if (!inited) {
        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx ++) {
            cudaCheck(cudaEventCreateWithFlags(&this->compl_event[layer_idx], cudaEventBlockingSync));
            cudaCheck(cudaEventCreateWithFlags(&this->internal_onload_event[layer_idx], cudaEventBlockingSync));
        }
        inited = true;
    }
}

void KVOnloadHandle::reset() {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        host_complete[layer_idx] = 0;
    }
}

void KVOnloadHandle::complete_host(int layer_idx, cudaStream_t stream) {
    cudaCheck(cudaEventRecord(compl_event[layer_idx], stream));
    {
        std::unique_lock<std::mutex> lock(mtx_);
        host_complete[layer_idx] = 1;
    }
    cv_.notify_one();
};

void KVOnloadHandle::wait_host(int layer_idx) {
    if (no_onload) return;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, layer_idx](){ return host_complete[layer_idx] == 1; });
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaCheck(cudaStreamWaitEvent(stream, compl_event[layer_idx], 0));
};


KVOffloadHandle::KVOffloadHandle(
    int num_layers,
    bool has_offload
)
: num_layers(num_layers)
, ready_event(std::vector<cudaEvent_t>(num_layers))
, host_ready(std::vector<std::atomic<int>>(num_layers, 0))
, no_offload(false) {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaCheck(cudaEventCreateWithFlags(&internal_gather_event[layer_idx], cudaEventBlockingSync));
        cudaCheck(cudaEventCreateWithFlags(&ready_event[layer_idx], cudaEventBlockingSync));
    }
}

KVOffloadHandle::~KVOffloadHandle() {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaCheck(cudaEventDestroy(&internal_gather_event[layer_idx]));
        cudaCheck(cudaEventDestroy(&ready_event[layer_idx]));
    }
}


void KVOffloadHandle::complete_host(int layer_idx, cudaStream_t stream,
                                    std::vector<std::pair< std::vector<void*>, std::vector<void*> >>&& chunks) {
    this->chunks = std::move(chunks);
    cudaCheck(cudaEventRecord(this->ready_event[layer_idx], stream));
    this->host_ready[layer_idx].store(1, std::memory_order_release);
}

bool KVOffloadHandle::try_wait_host(int layer_idx) {
    if (this->host_ready[layer_idx].load(std::memory_order_acquire) == 0) {
        return false;
    }
    if (cuEventQuery(this->ready_event[layer_idx]) == cudaSuccess) {
        return true;
    }
    return false;
}

HostKVStorageImpl::HostKVStorageImpl(
    int num_layers,
    int num_kv_heads,
    int kv_headdim,
    int num_tokens_per_page,
    int64_t num_tokens_per_chunk
)
    : num_layers(num_layers)
    , num_kv_heads(num_kv_heads)
    , kv_headdim(kv_headdim)
    , page_size(num_tokens_per_page)
    , chunk_size(num_tokens_per_chunk)
    , _uid_to_chunks(num_layers, std::unordered_map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>>())
{
    this->chunk_numel = num_tokens_per_chunk * 2 * num_kv_heads * kv_headdim;
    this->page_numel = 2 * page_size * num_kv_heads * kv_headdim;
    this->per_token_numel = 2 * num_kv_heads * kv_headdim;

    int dev_id = 0;
    int k_num_sms = 0;
    cudaCheck(cudaGetDevice(&dev_id));
    cudaCheck(cudaDeviceGetAttribute(&k_num_sms, cudaDevAttrMultiProcessorCount, dev_id));
    this->device_num_sms = k_num_sms;
};

HostKVStorageImpl::~HostKVStorageImpl()
{}

int64_t HostKVStorageImpl::get_kvdata_length(int64_t user_id) {
    auto it = _uid_to_length.find(user_id);
    if (it ==  _uid_to_length.end()) return 0;
    return it->second;
};

std::pair<std::vector<void*>, std::vector<int64_t>> HostKVStorageImpl::get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx) {
    // assert(this->get_kvdata_length(user_id) >= length);
    std::unique_lock<std::mutex> lock(host_kvcache_mutex_);

    std::vector<void*> chunk_ptrs;
    std::vector<int64_t> chunk_bytes;
    if (length == 0) {
        return std::make_pair(chunk_ptrs, chunk_bytes);
    }

    const auto &chunk_offsets = _uid_to_chunks[user_id].first;
    const auto &chunk_sizes = _uid_to_chunks[user_id].second;

    int64_t tokens_from_chunks = 0;
    for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
        void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(this->pinned_kvstorage_buffers[layer_idx]) + chunk_offsets[idx]);
        chunk_ptrs.push_back(src_ptr);
        if (tokens_from_chunks + chunk_sizes[idx] * this->num_tokens_per_page >= length) {
            int64_t last_chunk_tokens = length - tokens_from_chunks;
            chunk_bytes.push_back(last_chunk_tokens * this->per_token_numel * sizeof(uint16_t));
            break;
        }
        tokens_from_chunks += chunk_sizes[idx] * this->num_tokens_per_page;
        chunk_bytes.push_back(chunk_sizes[idx] * this->num_tokens_per_page * this->per_token_numel * sizeof(uint16_t));
    }
    // assert(tokens_from_chunks == length);
    return std::make_pair(chunk_ptrs, chunk_bytes);
};


std::vector<at::Tensor> HostKVStorageImpl::get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat) {
    int batch_size = user_ids.size();

    std::vector<int64_t> seqlens(batch_size, 0);
    int64_t total_seqlen = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        seqlens[seq_idx] = get_kvdata_length(user_ids[seq_idx]);
        total_seqlen += seqlens[seq_idx];
    }
    int64_t num_total_pages = total_seqlen / this->page_size;

    at::Tensor tensor_res = at::empty({
        this->num_layers, num_total_pages, 2, this->page_size, this->num_kv_heads, this->kv_headdim}, torch::kBFloat16);
    
    int64_t seqlen_offset = 0;
    uint16_t *raw_ptr = static_cast<uint16_t*>(tensor_res.data_ptr());


    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        uint16_t *seq_ptr = raw_ptr + (seqlen_offset / this->page_size) * tensor_res.stride(1);
        int num_chunks = seqlens[seq_idx] / this->chunk_size;

        for (int layer_idx = 0 ; layer_idx < this->num_layers; layer_idx++){
            auto [chunk_ptrs, chunk_bytes] = get_kvdata(user_ids[seq_idx], seqlens[seq_idx], layer_idx);
            for (int chk_idx = 0; chk_idx < num_chunks; chk_idx++) {
                std::memcpy(seq_ptr + layer_idx * tensor_res.stride(0) + chk_idx * this->chunk_numel, chunk_ptrs[chk_idx], chunk_bytes[chk_idx]);
            }
        }

        seqlen_offset += seqlens[seq_idx];
    }
    
    std::vector<at::Tensor> res({tensor_res});
    return res;
}

// void HostKVStorageImpl::init_random_kvdata(int64_t user_id, size_t num_tokens) {
//     if (_uid_to_length.find(user_id) != _uid_to_length.end()) return;
//     if (num_tokens == 0) return;
    
//     size_t num_chunks = ((num_tokens + this->chunk_size - 1) / this->chunk_size);

//     uint16_t *host_data_ptr = (uint16_t *)malloc(this->num_layers * num_chunks * this->chunk_numel * sizeof(uint16_t));

//     for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
//         _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
//     _uid_to_mempool[user_id] = std::vector<uintptr_t>();

//     for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
//         uint16_t* src_ptr = host_data_ptr + layer_idx * num_chunks * this->chunk_numel;
        
//         for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
//             _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->chunk_numel));
//         }
//     }
//     _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(host_data_ptr));
//     _uid_to_length[user_id] = num_chunks * this->chunk_size;
// }


void HostKVStorageImpl::onload_kvcache(
    at::Tensor onload_user_ids,  // on host
    const std::vector<const at::Tensor>& onload_page_indices_list, // on gpu
    KVOnloadHandle& onloadhandle) {
    const int batch_size = onload_user_ids.size(0);

    at::Device device = onload_page_indices_list[0].device();
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    c10::cuda::CUDAStream c10_onload_stream =
        c10::cuda::getStreamFromExternal(this->onload_stream, device.index());

    onloadhandle.init();
    // The onload_page_indices_list are views from paged_indices for inference attention, which 
    // contains more page_ids than required onload, so we need to do extra gpu memory allocation and memcpy.
    // Note: assume uid & page_ids without onload are already removed.
    // [[ contains allocation from torch gpu memory, and memcpy(concat) on gpu ]]
    c10::cuda::CUDAStreamGuard onload_stream_guard(c10_onload_stream);
    at::Tensor onload_page_indices = at::cat(onload_page_indices_list, 0);

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        // nvtx3::scoped_range r{"onload_layer_" + std::to_string(layer_idx)};
        char *gpu_onload_buffer = this->onload_gpu_buffers[layer_idx % 2];  // single layer for a max batch (multi-chunks)
        if (layer_idx >= 2) cudaCheck(cudaStreamWaitEvent(this->onload_stream, onloadhandle.compl_event[layer_idx - 2], 0));

        size_t bytes_offset_in_buffer = 0;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto [chunk_ptrs, chunk_bytes] = host_kv_mgr->get_kvdata(
                onload_user_ids[seq_idx], 
                onload_page_indices_list[seq_idx].size(0) * this->num_tokens_per_page, 
                layer_idx
            );
            for (int chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {
                cudaCheck(cudaMemcpyAsync(gpu_onload_buffer + bytes_offset_in_buffer,
                    chunk_ptrs[chunk_idx], chunk_bytes[chunk_idx], cudaMemcpyHostToDevice, this->onload_stream));
                bytes_offset_in_buffer += chunk_bytes[chunk_idx];
            }
        }
        cudaCheck(cudaEventRecord(onloadhandle.internal_onload_event[layer_idx], this->onload_stream));
        cudaCheck(cudaStreamWaitEvent(this->scatter_stream, onloadhandle.internal_onload_event[layer_idx], 0));
        scatter_paged_kvcache(this->gpu_cache_table[layer_idx], gpu_onload_buffer, onload_page_indices.data_ptr<int32_t>(), 
            this->num_kv_heads, this->kv_headdim, this->page_size, this->page_stride, this->k2v_stride, 
            this->num_kv_heads * this->kv_headdim, this->kv_headdim, onload_page_indices.size(0) * this->num_tokens_per_page,
            this->device_num_sms, this->scatter_stream);
        onloadhandle.complete_host(layer_idx, this->scatter_stream);
    }
}
// 32 * 8192 * 2 * 8 * 256 * 2 = 1342177280 bytes = 1.25 GB per layer

std::vector<std::vector<void*>> HostKVStorageImpl::get_empty_pinned_chunks(
    at::Tensor& offload_user_ids, 
    at::Tensor& offload_start_indices,
    const std::vector<int32_t>& offload_num_pages_list) {
    const auto batch_size = offload_user_ids.size(0);

    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> empty_pinned_chunks;
    
    for (auto idx = 0; idx < batch_size; idx++) {
        int64_t user_id = offload_user_ids[idx].item<int64_t>();
        int32_t start_pos = offload_start_indices[idx].item<int64_t>();
        int32_t offload_num_pages = offload_num_pages_list[idx];

        std::vector<int64_t> chunk_offsets;
        std::vector<int64_t> chunk_sizes;

        if (start_pos % num_tokens_per_chunk != 0)  {
            // if start_pos is not aligned to chunk size, reuse the last offloaded chunk.
            auto partial_num_pages = (start_pos % this->num_tokens_per_chunk) / this->num_tokens_per_page;
            auto chunk_offset = _uid_to_chunks[user_id].first.back();
            // assert();
            chunk_offset = chunk_offset + partial_num_pages * this->page_bytes;
            chunk_offsets.push_back(chunk_offset);
            chunk_sizes.push_back(this->num_pages_per_chunk - partial_num_pages);

            start_pos = int32_t(std::ceil(static_cast<float>(start_pos) / num_tokens_per_chunk) * num_tokens_per_chunk);
            offload_num_pages -= (this->num_pages_per_chunk - partial_num_pages);
        }

        int32_t padded_num_pages = static_cast<int32_t>(std::ceil(static_cast<float>(offload_num_pages) / this->num_tokens_per_chunk) * this->num_tokens_per_chunk);
        while (padded_num_pages > 0) {
            if (_empty_chunks.empty()) {
                // abort if no empty chunk available
                break;
            }
            auto [chunk_idx, chunk_size] = _empty_chunks.front();
            if (chunk_size > padded_num_pages) {
                _empty_chunks.front().first += padded_num_pages / this->num_pages_per_chunk;
                _empty_chunks.front().second -= padded_num_pages;
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_sizes.push_back(padded_num_pages);
                padded_num_pages = 0;
            } else {
                _empty_chunks.pop();
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_sizes.push_back(chunk_size);
                padded_num_pages -= chunk_size;
            }
        }
        _empty_sizes -= padded_num_pages;

        empty_pinned_chunks.push_back(std::make_pair(chunks, chunk_sizes));
    }

    return empty_pinned_chunks;
}


bool HostKVStorageImpl::offload_kvcache(
    at::Tensor offload_user_ids, // on host
    at::Tensor offload_start_indices,  // on host
    const std::vector<const at::Tensor>& offload_page_indices_list, // on host
    KVOffloadHandle& offloadhandle) {
    const auto batch_size = offload_user_ids.size(0);

    at::Device device = offload_page_indices_list[0].device();
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    c10::cuda::CUDAStream c10_gather_stream =
        c10::cuda::getStreamFromExternal(this->gather_stream, device.index());

    offloadhandle.init();
    std::vector<int64_t> filtered_user_ids;
    std::vector<const at::Tensor> filtered_page_indices_list;
    std::vector<int32_t> num_pages_list;
    // filter offload uids & page_ids
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int32_t start_index = offload_start_indices[seq_idx].item<int32_t>();
        const int32_t offloaded_length = this->_uid_to_length[seq_idx].item<int32_t>();
        if (start_index < offloaded_length) {
            // give a warning print
        } else if (start_index > offloaded_length) {
            // give an error print
            // uninit offloadhandle
            return false;
        }
        int32_t page_start = offloaded_length / this->num_tokens_per_page;
        int32_t page_end = start_index / this->num_tokens_per_page + offload_page_indices_list[seq_idx].size(0);
        if (page_end == page_start) continue;
        filtered_user_ids.push_back(offload_user_ids[seq_idx].item<int64_t>());
        filtered_page_indices_list.push_back(
            offload_page_indices_list[seq_idx].narrow(0, page_start, page_end - page_start));
        offloadhandle.pages.push_back(std::make_pair(page_start, page_end));
        num_pages_list.push_back(page_end - page_start);
    }
    c10::cuda::CUDAStreamGuard gather_stream_guard(c10_gather_stream);
    // [[ contains allocation from torch gpu memory, and memcpy(concat, h2d) ]]
    int64_t cat_dim = 0;
    for (const auto& t : filtered_page_indices_list) cat_dim += t.size(0);
    at::Tensor h_offload_page_indices = at::empty(
        {cat_dim},
        at::TensorOptions().device(torch::kCPU).dtype(torch::kInt32).pinned_memory(true)
    );
    at::cat_out(h_offload_page_indices, filtered_page_indices_list, 0);
    at::Tensor offload_page_indices = h_offload_page_indices.to(device, torch::kInt32);
    
    
    // Step 0. Find pinned buffers from the pool: 
    //      [[ In Use ]]: Eviction Policy based on user LRU, without pinned-to-unpinned memcpy
    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> empty_pinned_chunks = this->get_empty_pinned_chunks(
        offload_user_ids,
        offload_start_indices,
        num_pages_list,
    );

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        void *gpu_offload_buffer = this->offload_gpu_buffers[layer_idx % 2];  // single layer for a max batch (multi-chunks)
        if (layer_idx >= 2) cudaCheck(cudaStreamWaitEvent(this->gather_stream, offloadhandle.ready_event[layer_idx - 2], 0));
        // Step 1. callgatherkernel<<< , this->gather_stream>>>(gpu_offload_buffer, this->gpu_cache_table, offload_page_indices, offload_page_indptrs);
        gatter_paged_kvcache(gpu_offload_buffer, this->gpu_cache_table[layer_idx], offload_page_indices.data_ptr<int>(), 
            this->num_kv_heads, this->kv_headdim, this->page_size, this->page_stride, this->k2v_stride, 
            this->num_kv_heads * this->kv_headdim, this->kv_headdim, offload_page_indices.size(0),
            this->device_num_sms, this->scatter_stream);

        // Step 2. cudaEventRecord(offloadhandle.internal_gather_event, this->offload_stream);
        cudaCheck(cudaEventRecord(offloadhandle.internal_gather_event[layer_idx], this->gather_stream));
        // Step 3. cudaMemcpyAsync from GPU cache to pinned buffer
        cudaCheck(cudaStreamWaitEvent(this->offload_stream, offloadhandle.internal_gather_event[layer_idx], 0));
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto [chunk_offsets, chunk_sizes] = empty_pinned_chunks[seq_idx];
            size_t offset_in_buffer = 0;
            for (int idx = 0; idx < chunk_offsets.size(); idx++) {
                cudaCheck(cudaMemcpyAsync(this->pinned_kvstorage_buffers[layer_idx] + chunk_offsets[idx], gpu_offload_buffer + offset_in_buffer, chunk_sizes[idx] * this->page_bytes, cudaMemcpyDeviceToHost, this->offload_stream));
                offset_in_buffer += chunk_sizes[idx] * this->page_bytes;
            }
        }
        if (layer_idx < this->num_layers - 1) {
            offloadhandle.complete_host(layer_idx, this->offload_stream, std::move(offloadhandle.chunks));
        } else {
            offloadhandle.complete_host(layer_idx, this->offload_stream, std::move(empty_pinned_chunks));
        } 
    }
    return true;
}

bool HostKVStorageImpl::finish_offload(
    at::Tensor& offload_user_ids,
    KVOffloadHandle& offloadhandle) {

    for (int seq_idx = 0; seq_idx < offload_user_ids.size(0); seq_idx++) {
        int64_t user_id = offload_user_ids[seq_idx].item<int64_t>();
        auto [page_start, page_end] = offloadhandle.pages[seq_idx];
        if (page_start * this->page_size != _uid_to_length[user_id]) {
            // give a warning print
            return false;
        }

        auto [chunk_offsets, chunk_sizes] = offloadhandle.chunks[seq_idx];
        for (int idx = 0; idx < chunk_offsets.size(); idx++) {
            if (chunk_sizes[idx] < this->num_pages_per_chunk) continue;
            _uid_to_chunks[user_id].first.push_back(chunk_offsets[idx]);
            _uid_to_chunks[user_id].second.push_back(chunk_sizes[idx]);
        }
        _uid_to_length[user_id] = page_end * this->page_size;
    }

    return true;
}

bool HostKVStorageImpl::cancel_offload(
    KVOffloadHandle& offloadhandle) {
    // nop for canceling the launched kernels

    auto & chunks = offloadhandle.chunks;
    for (int seq_idx = 0; seq_idx < chunks.size(); seq_idx++) {
        auto [chunk_offsets, chunk_sizes] = chunks[seq_idx];
        if (chunk_offsets.size() == 0) continue;
        for (int idx = 0; idx < chunk_offsets.size(); idx++) {
            if (chunk_sizes[idx] < this->num_pages_per_chunk) continue;
            _empty_chunks.push(std::make_pair(chunk_offsets[idx] / this->unit_chunk_bytes, chunk_sizes[idx]));
            _empty_sizes += chunk_sizes[idx];
        }
    }

    return true;
}


}  // namespace kvcache