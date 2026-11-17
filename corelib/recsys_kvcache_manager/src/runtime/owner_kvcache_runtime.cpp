// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "owner_kvcache_runtime.h"

#include <ATen/cuda/CUDAContext.h>

#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include "cache_owner.h"
#include "kvcache_manager_context.h"
#include "kvcache_runtime_config.h"

namespace kvcache_manager {
namespace {
using State = OwnerTransferState;
std::shared_ptr<IKVCacheRuntime> raw() {
    TORCH_CHECK(cache_owner_worker_thread(), "raw cache access requires its owner thread");
    return KVCacheRuntimeContext::instance().manager();
}
at::Tensor empty_ids() {
    return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

class OwnerKVCacheRuntime final : public IKVCacheRuntime {
   public:
    OwnerKVCacheRuntime() {
        auto config = load_yaml_config(find_config_path());
        page_size_ = get_config_int(config, "tokens_per_page");
        auto device = get_config_int(config, "device_idx");
        anchor_ = empty_ids();
        owner_ = std::make_unique<CacheOwner>(device);
        owner_->initialize(anchor_);
        tables_ = owner_->sync([] { return raw()->get_kvcache_tables(); });
        // Tables were allocated on the owner stream. Publish the initialization
        // before callers can write them on their model stream.
        publish_metadata();
    }
    ~OwnerKVCacheRuntime() override {
        try {
            shutdown_runtime();
        } catch (...) { /* explicit shutdown reports errors */
        }
    }
    void shutdown_runtime() override {
        if (closed_) return;
        std::exception_ptr error;
        try {
            owner_->sync([&] {
                owner_->wait_request_completion();
                owner_->flush();
            });
        } catch (...) {
            error = std::current_exception();
        }
        owner_->sync([&] { owner_->shutdown(anchor_); }, true);
        closed_ = true;
        if (error) std::rethrow_exception(error);
    }
    void record_request_completion() override { owner_->record_request_completion(); }
    std::vector<at::Tensor> lookup_kvcache(at::Tensor ids, at::Tensor lengths) override {
        // Generic paged storage may evict or compact pages. Retire the preceding
        // consumer and transfers before its metadata can be reused. Fixed-slot
        // clients use admission/pins directly and do not take this drain path.
        auto result = owner_->sync([&] {
            owner_->wait_request_completion();
            owner_->flush();
            return raw()->lookup_kvcache(ids, lengths);
        });
        onboards_.clear();
        return result;
    }
    std::vector<at::Tensor> allocate_kvcache(at::Tensor ids, at::Tensor lengths, at::Tensor merged,
                                             at::Tensor host) override {
        auto result = owner_->sync([&] {
            owner_->wait_request_completion();
            owner_->flush();
            return raw()->allocate_kvcache(ids, lengths, merged, host);
        });
        publish_metadata();
        return result;
    }
    std::vector<at::Tensor> onboard_kvcache_launch(at::Tensor ids, at::Tensor lengths,
                                                   at::Tensor merged, at::Tensor host,
                                                   at::Tensor start, at::Tensor gpu,
                                                   at::Tensor& tasks, at::Tensor indices,
                                                   at::Tensor indptr) override {
        auto r = std::make_shared<State>();
        r->ids = ids.clone();
        r->lengths = lengths.clone();
        r->count = ids.numel();
        r->found.resize(7);
        r->found[1] = merged.clone();
        r->found[2] = start.clone();
        r->found[3] = gpu.clone();
        r->found[5] = host.clone();
        r->found[6] = tasks.clone();
        r->allocation.resize(3);
        r->allocation[0] = indices;
        r->allocation[2] = indptr;
        r->pools = tables_;
        auto handles = at::full_like(tasks, -1);
        bool full_gpu_hit = r->count > 0;
        for (int64_t i = 0; i < r->count; ++i) {
            const auto m = merged.data_ptr<int32_t>()[i], h = host.data_ptr<int32_t>()[i];
            const auto s = start.data_ptr<int32_t>()[i], g = gpu.data_ptr<int32_t>()[i];
            full_gpu_hit &= h == 0 && s == 0 && g == lengths.data_ptr<int64_t>()[i] &&
                            m == lengths.data_ptr<int64_t>()[i];
            if (m && h && (h > s + g || s > 0)) ++r->cpu_hits;
        }
        // Keep the standard slot mapping output; moving the network launch to
        // the owner does not change its values or partial-prefix semantics.
        std::vector<at::Tensor> mappings;
        if (full_gpu_hit) {
            for (int64_t i = 0; i < r->count; ++i) mappings.push_back(empty_ids());
        } else {
            mappings = owner_->sync([&] {
                auto cpu_indices = indices.cpu();
                auto cpu_indptr = indptr.cpu();
                std::vector<at::Tensor> result;
                auto offsets = at::arange(page_size_, empty_ids().options());
                for (int64_t i = 0; i < r->count; ++i) {
                    auto begin = cpu_indptr[i].item<int64_t>();
                    auto end = cpu_indptr[i + 1].item<int64_t>();
                    auto pages = cpu_indices.slice(0, begin, end).to(at::kLong);
                    result.push_back((pages.unsqueeze(1) * page_size_ + offsets.unsqueeze(0))
                                         .reshape({-1})
                                         .slice(0, 0, lengths.data_ptr<int64_t>()[i]));
                }
                return result;
            });
        }
        if (r->cpu_hits) {
            auto handle = next_handle_++;
            onboards_[handle] = r;
            // One logical handle represents a batched launch. The owner keeps
            // FlexKV's actual/merged task IDs private.
            handles.data_ptr<int64_t>()[0] = handle;
            owner_->post([this, r] { owner_->onboard(r); });
        }
        tasks = handles;
        return mappings;
    }
    void onboard_kvcache_wait(at::Tensor tasks) override {
        for_each_onboard(tasks, [&](const std::shared_ptr<State>& r) {
            OwnerWaitGuard wait_guard;
            std::unique_lock<std::mutex> lock(r->mutex);
            bool ready = r->cv.wait_for(lock, std::chrono::seconds(30),
                                        [&] { return r->onboard_complete || r->error; });
            TORCH_CHECK(ready, "cache owner onboarding timed out");
            if (r->error) std::rethrow_exception(r->error);
        });
        owner_->check();
    }
    bool onboard_kvcache_poll(at::Tensor tasks) override {
        bool ready = true;
        for_each_onboard(tasks, [&](const std::shared_ptr<State>& r) {
            std::lock_guard<std::mutex> lock(r->mutex);
            if (r->error) std::rethrow_exception(r->error);
            ready &= r->onboard_complete;
        });
        owner_->check();
        return ready;
    }
    bool fixed_allocation_available(at::Tensor ids) override {
        return owner_->sync([&] { return raw()->fixed_allocation_available(ids); });
    }
    bool try_resident_slots(at::Tensor ids, int64_t length, int64_t banks,
                            at::Tensor slots) override {
        return owner_->sync([&] { return raw()->try_resident_slots(ids, length, banks, slots); });
    }
    at::Tensor offload_kvcache_launch(at::Tensor ids, at::Tensor lengths, at::Tensor merged,
                                      at::Tensor host, at::Tensor start, at::Tensor gpu,
                                      at::Tensor indices, at::Tensor indptr,
                                      std::vector<at::Tensor> mappings) override {
        auto r = std::make_shared<State>();
        r->ids = ids.clone();
        r->lengths = lengths.clone();
        r->count = ids.numel();
        r->found.resize(7);
        r->found[1] = merged.clone();
        r->found[2] = start.clone();
        r->found[3] = gpu.clone();
        r->found[5] = host.clone();
        r->allocation.resize(3);
        r->allocation[0] = indices;
        r->allocation[2] = indptr;
        r->pools = tables_;
        auto handles = at::empty_like(ids);
        for (int64_t i = 0; i < ids.numel(); ++i) handles.data_ptr<int64_t>()[i] = next_handle_++;
        r->completion_ids = handles;
        // Honor caller-provided mappings, including a custom partial-prefix
        // layout. Clone CPU metadata because publication outlives this call.
        for (const auto& mapping : mappings) r->slot_mappings.push_back(mapping.clone());
        // Never let an asynchronous copy observe unfinished append kernels.
        // The owner queries this event without blocking the model thread.
        r->done.record(at::cuda::getCurrentCUDAStream());
        owner_->submit_offload(r);
        owner_->record_request_completion();
        return handles;
    }
    at::Tensor offload_kvcache_reap_completed() override {
        return owner_->sync([&] { return owner_->take_reaped(); });
    }
    std::vector<at::Tensor> get_kvcache_tables() override { return tables_; }
    int64_t get_num_offload_tasks() override {
        return owner_->sync([&] { return owner_->pending_offloads(); });
    }
    void evict_kvcache(at::Tensor ids, bool gpu_only) override {
        owner_->sync([&] {
            owner_->wait_request_completion();
            owner_->flush();
            raw()->evict_kvcache(ids, gpu_only);
        });
    }

   private:
    template <class F>
    void for_each_onboard(at::Tensor tasks, F fn) {
        std::unordered_set<int64_t> seen;
        for (int64_t i = 0; i < tasks.numel(); ++i) {
            auto handle = tasks.data_ptr<int64_t>()[i];
            if (handle < 0 || !seen.insert(handle).second) continue;
            auto it = onboards_.find(handle);
            TORCH_CHECK(it != onboards_.end(), "unknown or retired owner onboard task ", handle);
            fn(it->second);
        }
    }
    void publish_metadata() {
        auto event = std::make_shared<at::cuda::CUDAEvent>();
        owner_->sync([&] { event->record(at::cuda::getCurrentCUDAStream()); });
        event->block(at::cuda::getCurrentCUDAStream());
    }
    std::unique_ptr<CacheOwner> owner_;
    at::Tensor anchor_;
    std::vector<at::Tensor> tables_;
    std::unordered_map<int64_t, std::shared_ptr<State>> onboards_;
    int64_t page_size_ = 0, next_handle_ = int64_t{1} << 50;
    bool closed_ = false;
};
}  // namespace
std::shared_ptr<IKVCacheRuntime> make_owner_kvcache_runtime() {
    return std::make_shared<OwnerKVCacheRuntime>();
}
}  // namespace kvcache_manager
