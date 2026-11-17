#pragma once

#include <memory>
#include <ATen/ATen.h>

namespace kvcache_manager {

class IKVCacheRuntime {
public:
    IKVCacheRuntime() = default;
    virtual ~IKVCacheRuntime() = default;
    virtual void record_request_completion() {}
    virtual void shutdown_runtime() {}

    virtual std::vector<at::Tensor> lookup_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens
    ) = 0;
    // virtual std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> 
    virtual std::vector<at::Tensor> allocate_kvcache(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths
    ) = 0;
    virtual std::vector<at::Tensor> onboard_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor& task_ids,  // from `get_match` in lookup and reuse in onboarding only, returned as masked
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr
    ) = 0;
    virtual void onboard_kvcache_wait(at::Tensor user_ids) = 0;
    virtual bool onboard_kvcache_poll(at::Tensor task_ids) = 0;
    virtual bool fixed_allocation_available(at::Tensor user_ids) = 0;
    // Optional fixed-slot fast path. The serialized caller protects active
    // inference storage through completion; this is not a concurrent lease API.
    virtual bool try_resident_slots(at::Tensor user_ids, int64_t length,
                                    int64_t banks, at::Tensor cpu_slots) {
        return false;
    }
    virtual at::Tensor offload_kvcache_launch(
        at::Tensor user_ids,
        at::Tensor seqlens,
        at::Tensor merged_cached_lengths,
        at::Tensor host_cached_lengths,
        at::Tensor gpu_cached_startpos,
        at::Tensor gpu_cached_lengths,
        at::Tensor kv_page_indices,
        at::Tensor kv_page_indptr,
        std::vector<at::Tensor> slot_mappings
    ) = 0;
    virtual at::Tensor offload_kvcache_reap_completed() = 0;
    virtual std::vector<at::Tensor> get_kvcache_tables() = 0;
    virtual int64_t get_num_offload_tasks() = 0;

public:
    virtual void evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) = 0;
};

} // namespace kvcache_manager

