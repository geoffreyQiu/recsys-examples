// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "owner_wait_guard.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>

#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace kvcache_manager {

// Shared transfer lifetime. Model adapters may derive their own request state;
// cache-layout-specific counters and graph tickets do not belong to the owner.
struct OwnerTransferState {
    at::Tensor ids, input_ids, lengths, tasks;
    std::vector<at::Tensor> found, allocation, pools;
    int64_t count = 0, cpu_hits = 0;
    bool onboard_complete = false, onboard_launched = false;
    bool compact_resident = false;
    std::mutex mutex;
    std::condition_variable cv;
    std::exception_ptr error;
    at::cuda::CUDAEvent done, metadata;
    // Optional externally returned opaque IDs for asynchronous offload/reap.
    at::Tensor completion_ids;
    // Optional standard-op slot mappings. Empty means reconstruct from pages.
    std::vector<at::Tensor> slot_mappings;
};

class __attribute__((visibility("default"))) CacheOwner {
   public:
    explicit CacheOwner(int device, size_t capacity = 64);
    ~CacheOwner();
    CacheOwner(const CacheOwner&) = delete;
    CacheOwner& operator=(const CacheOwner&) = delete;
    void post(std::function<void()> command, bool allow_error = false);
    template <class F>
    auto sync(F fn, bool allow_error = false) -> decltype(fn()) {
        using Result = decltype(fn());
        auto task = std::make_shared<std::packaged_task<Result()>>(
            [this, fn = std::move(fn), allow_error]() mutable {
                if (!allow_error) check();
                return fn();
            });
        auto future = task->get_future();
        post([task] { (*task)(); }, allow_error);
        OwnerWaitGuard wait_guard;
        return future.get();
    }
    void initialize(at::Tensor anchor);
    void check();
    void onboard(const std::shared_ptr<OwnerTransferState>& request);
    void submit_offload(std::shared_ptr<OwnerTransferState> request);
    // The following operations execute on the owner thread (usually via sync).
    void admission(at::Tensor ids);
    void flush();
    at::Tensor stats();
    void shutdown(at::Tensor anchor);
    int64_t pending_offloads();
    at::Tensor take_reaped();
    void wait_request_completion();
    // Called on the model thread after its last GPU consumer/producer.
    void record_request_completion();

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// The owner worker must always use the raw manager, never construct a nested
// owner facade through the thread-local Torch binding context.
__attribute__((visibility("default"))) bool cache_owner_worker_thread();

}  // namespace kvcache_manager
