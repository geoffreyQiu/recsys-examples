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

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream);

namespace kvcache {

GPUKVCacheMangerImpl::GPUKVCacheMangerImpl(
    int num_layers,
    int num_kv_heads,
    int kv_headdim,
    int num_tokens_per_page,
    int num_primary_cache_pages,
    int num_onload_buffer_pages,
    int num_reserved_buffer_pages,
    int num_tokens_per_chunk,
    int max_num_sequences,
    int max_sequence_length,
    at::Tensor cache_table_tensor,
    HostKVStorageImpl& host_kv_mgr,
    size_t max_queued_offload_tokens,
    int onload_buffer_chunks,
    int offload_buffer_chunks,
    int num_memcpy_workers,
    bool enable_nvcomp)
    : num_layers(num_layers)
    , num_kv_heads(num_kv_heads)
    , kv_headdim(kv_headdim)
    , num_tokens_per_page(num_tokens_per_page)
    , num_primary_cache_pages(num_primary_cache_pages)
    , num_onload_buffer_pages(num_onload_buffer_pages)
    , num_reserved_buffer_pages(num_reserved_buffer_pages)
    , num_tokens_per_chunk(num_tokens_per_chunk)
    , max_num_sequences(max_num_sequences)
    , max_sequence_length(max_sequence_length)
    , queued_offload_limits(max_queued_offload_tokens)
    , num_onload_device_chunks(onload_buffer_chunks)
    , num_offload_device_chunks(offload_buffer_chunks)
    , enable_nvcomp(enable_nvcomp)
    , compressor(enable_nvcomp ? 8 : 0, host_kv_mgr.chunk_numel, host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , onload_pin_buffer(host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , offload_pin_buffer(host_kv_mgr.chunk_numel * sizeof(uint16_t))
    , onload_memcpy_workers(num_memcpy_workers)
    , offload_memcpy_workers(num_memcpy_workers)
    , cache_table(static_cast<uint16_t*>(cache_table_tensor.data_ptr()))
    , device(cache_table_tensor.device())
{
    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

    for (int page_id = 0; page_id < num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);

    page_stride = 2 * num_tokens_per_page * num_kv_heads * kv_headdim;
    k2v_stride = num_tokens_per_page * num_kv_heads * kv_headdim;
    layer_stride = (num_primary_cache_pages + num_onload_buffer_pages) * page_stride;

    per_token_kv_stride = 2 * num_kv_heads * kv_headdim;

    cudaCheck(cudaStreamCreateWithFlags(&worker_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&onload_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&offload_stream, cudaStreamNonBlocking));

    this->host_kv_mgr = &host_kv_mgr;

    cudaCheck(cudaMalloc((void**)&onload_device_buffers, this->num_onload_device_chunks * host_kv_mgr.chunk_numel * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void**)&offload_device_buffers, this->num_offload_device_chunks * host_kv_mgr.chunk_numel * sizeof(uint16_t)));
    compressor.set_compress_input_buffer_ptrs(
        reinterpret_cast<char*>(offload_device_buffers), this->num_offload_device_chunks);

    this->terminate_ = false;

    this->queued_offload_tokens = 0;
    this->queued_offload_limits = max_queued_offload_tokens;
    this->offload_busy_.store(false);
    this->offload_worker = std::thread(&GPUKVCacheMangerImpl::offload_loop, this);
};

GPUKVCacheMangerImpl::~GPUKVCacheMangerImpl() {
    {
        std::unique_lock<std::mutex> lock(offload_task_mutex_);
        this->terminate_ = true;
    }
    offload_task_cv_.notify_all();

    this->offload_worker.join();

    cudaFree(offload_device_buffers);
    cudaFree(onload_device_buffers);
}

int64_t GPUKVCacheMangerImpl::getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids) {
    while (true) {
        int num_offloading_uids = 0;
        {
            std::unique_lock<std::mutex> lock(offload_freezed_uids_mtx_);
            num_offloading_uids = offload_freezed_uids_.size();

            for (auto it = std::rbegin(_lru_list); it != std::rend(_lru_list); ++it) {
                if (offload_freezed_uids_.find((int64_t)*it) != offload_freezed_uids_.end())
                    continue;
                if (extra_freezed_uids.find((int64_t)*it) != extra_freezed_uids.end())
                    continue;
                return *it;
            }
        }
        if (num_offloading_uids == 0) assert(false);
        
        std::this_thread::yield();
    }

    return _lru_list.back();
};

std::vector<int32_t>& GPUKVCacheMangerImpl::alloc_single_sequence(int64_t uid, int new_total_length, int host_cached_startpos, int host_cached_length, std::unordered_set<int64_t> freezed_uids) {
    (void)host_cached_startpos;  // assume to be zero in current implementation.

    int num_total_pages = int((new_total_length + this->num_tokens_per_page - 1) / this->num_tokens_per_page);  // duplicated computation

    int cur_cached_start = 0;
    int cur_cached_len = 0;
    bool found_in_gpu_cache = retain(uid);
    if (found_in_gpu_cache) {  // duplicated lookup
        cur_cached_start = _uid_to_paged_cache_startpos[uid];
        cur_cached_len = _uid_to_paged_cache_length[uid];
    }

    // 1. allocate pages for host cached data
    int num_onload_pages = 0;
    if (cur_cached_len == 0) {
        num_onload_pages = int((host_cached_length + this->num_tokens_per_page - 1) / this->num_tokens_per_page);
        // Note: We would allow host_cached_length not aligned with this->num_tokens_per_page.
        //       And in this case, num_cur_pages (below) will be 0. Thus, num_append_pages is correct.
    } else if (cur_cached_start > 0) {
        // cur_cached_start should align with this->num_tokens_per_page;
        num_onload_pages = int(cur_cached_start / this->num_tokens_per_page);
    } /* else {
        assert(cur_cached_start == 0 && cur_cached_len > 0);
        num_onload_pages = 0;
    } */
    
    // 2. allocate pages for new data to be appended
    int num_cur_pages = (cur_cached_len + this->num_tokens_per_page - 1) / this->num_tokens_per_page;
    // assert((num_cur_pages == 0 && _uid_to_page_id.find(uid) == _uid_to_page_id.end()) ||
    //        num_cur_pages == _uid_to_page_id[uid].size()                                  );
    int num_append_pages = num_total_pages - num_onload_pages - num_cur_pages;
    
    int num_required_pages = num_onload_pages + num_append_pages;
    while ((size_t)num_required_pages > _empty_pages.size()) {
        int64_t uid_to_evict = getUIdToEvict(freezed_uids);
        evict(uid_to_evict);
    }

    std::vector<int32_t>& page_ids(num_total_pages);
    for (int i = 0; i < num_offload_pages; i++) {
        page_ids[i] = _empty_pages.front();
        _empty_pages.pop();
    }
    for (int i = num_offload_pages; i < num_offload_pages + num_cur_pages; i++) {
        page_ids[i] = _uid_to_page_id[uid][i - num_offload_pages];
    }
    for (int i = num_offload_pages + num_cur_pages; i < num_total_pages; i++) {
        page_ids[i] = _empty_pages.front();
        _empty_pages.pop();
    }
    _uid_to_page_id[uid] = page_ids;
    _uid_to_paged_cache_startpos[uid] = 0;
    _uid_to_paged_cache_length[uid] = new_total_length;

    return page_ids;
};

std::vector<std::vector<int32_t>> GPUKVCacheMangerImpl::lookup(at::Tensor uids) {
    int batch_size = uids.size(0);
    int *user_ids_ptr = uids.data_ptr<int>();
    std::vector<int32_t> cached_startpos(batch_size);
    std::vector<int32_t> cached_lengths(batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        if (_uid_to_paged_cache_startpos.find(uid) != _uid_to_paged_cache_startpos.end()) {
            cached_startpos[seq_idx] = _uid_to_paged_cache_startpos[uid];
            cached_lengths[seq_idx] = _uid_to_paged_cache_startpos[uid] + _uid_to_paged_cache_length[uid];
        } else if (_uid_to_offloaded_length.find(uid) != _uid_to_offloaded_length.end())
            ;  // should not get this; debug print and fail
        else {
            cached_startpos[seq_idx] = 0;
            cached_lengths[seq_idx] = 0;
        }
    }
    return {cached_startpos, cached_lengths};
};
    
void GPUKVCacheMangerImpl::evict(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    assert(_lru_lookup_table.end() != tableIt);
    // if (_lru_lookup_table.end() != tableIt) {
        _lru_list.erase(tableIt->second);
        _lru_lookup_table.erase(tableIt);
        // assert(_uid_to_page_id[uid].size() > 0);

        for (auto page_id : _uid_to_page_id[uid]) {
            _empty_pages.push(page_id);
        }

        _uid_to_page_id.erase(uid);
        _uid_to_paged_cache_startpos.erase(uid);
        _uid_to_paged_cache_length.erase(uid);
    // }
};

void GPUKVCacheMangerImpl::evict_all()
{
    std::queue<int64_t> empty_pages;
    std::swap(_empty_pages, empty_pages);
    _lru_list.clear();
    _lru_lookup_table.clear();
    _uid_to_page_id.clear();
    _uid_to_paged_cache_startpos.clear();
    _uid_to_paged_cache_length.clear();

    for (int page_id = 0; page_id < this->num_primary_cache_pages; page_id++)
        _empty_pages.push(page_id);

    this->host_kv_mgr->_uid_to_chunk_id.clear();
    this->host_kv_mgr->_uid_to_length.clear();
    // for (auto p : this->host_kv_mgr->_uid_to_mempool) {
    //     for (auto ptr : p.second) {
    //         free(reinterpret_cast<uint16_t*>(ptr));
    //     }
    // }
    this->host_kv_mgr->_uid_to_mempool.clear();
};

void GPUKVCacheMangerImpl::invalid(int64_t uid) {
    auto const tableIt = _lru_lookup_table.find(uid);
    if (_lru_lookup_table.end() != tableIt) {
        _lru_list.erase(tableIt->second);
        _lru_lookup_table.erase(tableIt);

        for (auto page_id : _uid_to_page_id[uid]) {
            _empty_pages.push(page_id);
        }

        _uid_to_page_id.erase(uid);
        _uid_to_paged_cache_startpos.erase(uid);
        _uid_to_paged_cache_length.erase(uid);
        _uid_to_offloaded_length.erase(uid);
    }
};

bool GPUKVCacheMangerImpl::retain(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    bool found = (_lru_lookup_table.end() != tableIt);
    if (found) {
        _lru_list.erase(tableIt->second);
    }
    _lru_list.push_front(uid);
    _lru_lookup_table[uid] = _lru_list.begin();
    return found;
};


void GPUKVCacheMangerImpl::allocate(
    at::Tensor user_ids,
    std::vector<int64_t>& total_hist_lens,  // all histo w/o candi
    at::Tensor host_cached_lengths,
    at::Tensor page_ids_gpu_buffer,
    at::Tensor metadata_gpu_buffer) {

    // at::Tensor offload_page_ids_gpu_buffer,  // ? offload
    // at::Tensor offload_uids_buffer,  // ? offload

    const c10::cuda::OptionalCUDAGuard device_guard(this->device);

    void *metadata_host_buffer = metadata_host_buffers[0];

    int batch_size = user_ids.size(0);
    int64_t *user_ids_ptr = static_cast<int64_t*>(user_ids.data_ptr());
    int32_t *host_cached_lengths_ptr = static_cast<int32_t*>(host_cached_lengths.data_ptr());

    // duplicated lookup. necessary if serving with multiple inference instances in future, and need a lock above.
    std::vector<int32_t> old_history_lengths = this->lookup(user_ids)[1];

    std::vector<int> page_indices;
    std::vector<int> offload_page_ids;
    int64_t *offload_user_ids = static_cast<int64_t*>(offload_uids_buffer.data_ptr());

    int *host_bufptr = static_cast<int*>(metadata_host_buffer);

    int *page_indptr = host_bufptr + 0;
    int *last_page_len = host_bufptr + batch_size + 1;
    int *total_history_lengths = host_bufptr + batch_size * 2 + 1;
    int *total_history_offsets = host_bufptr + batch_size * 3 + 1;
    int *new_history_nnz_cuda = host_bufptr + batch_size * 4 + 2;
    int *new_history_offsets = host_bufptr + batch_size * 4 + 3;
    // === ^ GPU === v Host ===
    int *new_offload_startpos = host_bufptr + batch_size * 5 + 4;
    int *new_offload_lengths = host_bufptr + batch_size * 6 + 4;

    int *num_page_ids = host_bufptr + batch_size * 7 + 4;
    int *num_offload_page_ids = host_bufptr + batch_size * 7 + 5;
    int *num_offload_user_ids = host_bufptr + batch_size * 7 + 6;

    size_t onload_page_offset = this->num_primary_cache_pages;
    size_t num_offload_uids = 0;
    size_t num_offload_pages = 0;

    page_indptr[0] = 0;
    total_history_offsets[0] = 0;
    new_history_offsets[0] = 0;

    const std::unordered_set<int64_t> freezed_uids(user_ids_ptr, user_ids_ptr + batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids_ptr[seq_idx];
        int total_history_length = total_hist_lens[seq_idx];

        // [attention/get metadata]; changed due to no separated onload block
        std::vector<int32_t>& page_ids = this->alloc_single_sequence(uid, total_history_length, 0, host_cached_lengths_ptr[seq_idx], freezed_uids);
        page_indices.insert(page_indices.end(), page_ids.begin(), page_ids.end());
        page_indptr[seq_idx + 1] = page_indptr[seq_idx] + page_ids.size();
        last_page_len[seq_idx] = this->_uid_to_paged_cache_length[uid] % this->num_tokens_per_page;  // NOT duplicated lookup, updated
        // int gpu_cache_startpos = this->_uid_to_paged_cache_startpos[uid];  // #1 -- NOT duplicated lookup, updated ; #2 -- assume to be zero after alloc and during inference

        // [history metadata]
        total_history_lengths[seq_idx] = total_history_length;  // used for [appending/put metadata]
        total_history_offsets[seq_idx + 1] = total_history_offsets[seq_idx] + total_history_length;  // used as [attention metadata]: k/v seqlen_offsets (need adding jagged_data.num_candidates_offsets)
        new_history_offsets[seq_idx + 1] = new_history_offsets[seq_idx] + total_history_length - old_history_lengths[seq_idx];  // used for [appending/put metadata]
        // old_cached_lengths;  // used in [position encoding metadata]; from lookup results

        // offload start (?)
            //    auto offloaded_length = 0;
            //    auto chunked_length = total_history_length - total_history_length % this->num_tokens_per_chunk;
            //    if (this->_uid_to_offloaded_length.find(uid) != this->_uid_to_offloaded_length.end())
            //        offloaded_length = this->_uid_to_offloaded_length[uid];
            //    {
            //        std::unique_lock<std::mutex> lock(this->queued_offload_lastpos_mutex_);
            //        if (this->queued_offload_lastpos.find(uid) != this->queued_offload_lastpos.end()) {
            //            offloaded_length = this->queued_offload_lastpos[uid];
            //        }
            //        if (total_history_length - offloaded_length >= this->num_tokens_per_chunk) {
            //            this->queued_offload_lastpos[uid] = chunked_length;
            //        }
            //    }
            //    if (total_history_length - offloaded_length >= this->num_tokens_per_chunk) {
            //        auto new_offload_page_start = (offloaded_length - gpu_cache_startpos) / this->num_tokens_per_page;
            //        offload_user_ids[num_offload_uids] = uid;
            //        new_offload_startpos[num_offload_uids] = offloaded_length;
            //        new_offload_lengths[num_offload_uids] = chunked_length - offloaded_length;
            //        auto num_pages = new_offload_lengths[num_offload_uids] / this->num_tokens_per_page;
            //        offload_page_ids.insert(offload_page_ids.end(), page_ids.begin() + new_offload_page_start, page_ids.begin() + new_offload_page_start + num_pages);
            //        
            //        num_offload_uids += 1;
            //        num_offload_pages += num_pages;
            //    }
        // offload end (?)
    }

    cudaCheck(cudaMemcpyAsync(page_ids_gpu_buffer.data_ptr(), page_indices.data(), page_indptr[batch_size] * sizeof(int32_t), cudaMemcpyHostToDevice, this->worker_stream));
    // offload related
        //    cudaCheck(cudaMemcpyAsync(offload_page_ids_gpu_buffer.data_ptr(), offload_page_ids.data(), num_offload_pages * sizeof(int32_t), cudaMemcpyHostToDevice, this->worker_stream));
    // offload related end

    // [appending/put metadata; for cudagraph only]
    auto new_tokens = new_history_offsets[batch_size];
    *new_history_nnz_cuda = new_tokens;

    size_t host_buffer_d2h_size = (batch_size * 5 + 4) * sizeof(int32_t);
    cudaCheck(cudaMemcpyAsync(metadata_gpu_buffer.data_ptr(), metadata_host_buffer.data_ptr(), host_buffer_d2h_size, cudaMemcpyHostToDevice, this->worker_stream));
    
    // [appending/put metadata]
    int *gpu_bufptr = static_cast<int*>(metadata_gpu_buffer.data_ptr());
    int *total_history_lengths_dev = gpu_bufptr + batch_size * 2 + 1;
    int *new_history_offsets_dev = gpu_bufptr + batch_size * 4 + 3;
    int *batch_indices_dev = gpu_bufptr + batch_size * 5 + 4;
    int *position_dev = gpu_bufptr + batch_size * 5 + 4 + new_tokens;

    GetPagedBatchIndicesPositions(
        batch_size,
        new_history_offsets_dev,  // new_history_offsets
        total_history_lengths_dev,  // total_history_lengths
        batch_indices_dev,
        position_dev,
        this->worker_stream
    );
    cudaCheck(cudaStreamSynchronize(this->worker_stream));
}

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> GPUKVCacheMangerImpl::acquire_offload_pages(
    at::Tensor& user_ids,
    at::Tensor& offloaded_lengths) {
    const auto batch_size = user_ids.size(0);

    std::vector<int64_t> offload_user_ids;
    std::vector<int32_t> offload_startpos;
    std::vector<int32_t> offload_page_ids;
    std::vector<at::Tensor> offload_page_ids_list;

    int num_pages_to_offload = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        int64_t offloaded_length = offloaded_lengths[seq_idx].item<int64_t>();

        if (this->_uid_to_offloaded_length.find(uid) == this->_uid_to_offloaded_length.end()) {
            this->_uid_to_offloaded_length[uid] = 0;
        }

        if (this->_uid_to_offloaded_length[uid] != offloaded_length) {
            if (this->_uid_to_offloaded_length[uid] > offloaded_length) {
                // should not happen; debug print and fail
                assert(false);
            }
            this->_uid_to_offloaded_length[uid] = offloaded_length;
        }

        int num_offloaded_pages = int(offloaded_length / this->num_tokens_per_page);
        if (this->_uid_to_paged_cache_length[uid] - offloaded_length >= this->num_tokens_per_chunk) {
            offload_user_ids.push_back(uid);
            offload_startpos.push_back(offloaded_length);
            offload_page_ids_list.push_back(
                at::from_blob(
                    this->_uid_to_page_id[uid].data() + num_offloaded_pages, 
                    {static_cast<int64_t>(this->_uid_to_page_id[uid].size() - num_offloaded_pages)}, 
                    at::dtype(at::kInt32)
                )
            )
        }
    }

    for (auto uid : this->_lru_list) {
        if (this->num_offloaded_pages + num_pages_to_offload + this->empty_pages.size() > this->num_primary_cache_pages - this->num_buffer_pages) {
            break;
        }

        int offloaded_length = this->_uid_to_offloaded_length[uid];
        int num_offloaded_pages = int(offloaded_length / this->num_tokens_per_page);
    
        if (offloaded_length < this->_uid_to_paged_cache_length[uid]) {
            offload_user_ids.push_back(uid);
            offload_startpos.push_back(offloaded_length);
            offload_page_ids_list.push_back(
                at::from_blob(
                    this->_uid_to_page_id[uid].data() + num_offloaded_pages, 
                    {static_cast<int64_t>(this->_uid_to_page_id[uid].size() - num_offloaded_pages)}, 
                    at::dtype(at::kInt32)
                )
            )
        }
    }
    return std::make_tuple(
        at::from_blob(offload_user_ids.data(), {static_cast<int64_t>(offload_user_ids.size())}, at::dtype(at::kLong)),
        at::from_blob(offload_startpos.data(), {static_cast<int64_t>(offload_startpos.size())}, at::dtype(at::kInt32)),
        offload_page_ids_list,
    );
}


void GPUKVCacheMangerImpl::release_offload_pages(
    at::Tensor user_ids,
    std::vector<int64_t>& total_hist_lens,  // all histo w/o candi
    at::Tensor host_cached_lengths,
    at::Tensor page_ids_gpu_buffer,
    at::Tensor metadata_gpu_buffer) {
}


}  // namespace kvcache