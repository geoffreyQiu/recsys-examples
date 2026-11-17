// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "cache_owner.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <chrono>
#include <deque>
#include <thread>
#include <unordered_map>

namespace kvcache_manager {
namespace {
thread_local bool owner_worker_thread = false;
using Request = OwnerTransferState;
using Stack = std::vector<c10::IValue>;
Stack call(const char* name, Stack args) {
    c10::Dispatcher::singleton().findSchemaOrThrow(name, "").callBoxed(&args);
    return args;
}
at::Tensor ints(const std::vector<int64_t>& values) {
    auto result = at::empty({static_cast<int64_t>(values.size())},
                            at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    std::copy(values.begin(), values.end(), result.data_ptr<int64_t>());
    return result;
}
}  // namespace
bool cache_owner_worker_thread() { return owner_worker_thread; }
class CacheOwner::Impl {
   public:
    explicit Impl(int device, size_t capacity = 64) : capacity_(capacity) {
        thread_ = std::thread([this, device] {
            owner_worker_thread = true;
            c10::cuda::CUDAGuard device_guard(device);
            auto stream = at::cuda::getStreamFromPool(false, device);
            c10::cuda::CUDAStreamGuard stream_guard(stream);
            c10::InferenceMode inference;
            for (;;) {
                std::function<void()> command;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (queue_.empty() && !stop_) {
                        if (has_work_)
                            cv_.wait_for(lock, std::chrono::microseconds(50));
                        else
                            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                    }
                    if (stop_ && queue_.empty()) break;
                    if (!queue_.empty()) {
                        command = std::move(queue_.front());
                        queue_.pop_front();
                        space_.notify_all();
                    }
                }
                try {
                    if (!failure()) progress();
                } catch (...) {
                    fail(std::current_exception());
                }
                try {
                    if (command) command();
                } catch (...) {
                    fail(std::current_exception());
                }
            }
        });
    }
    ~Impl() {
        OwnerWaitGuard wait_guard;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            cv_.notify_all();
            space_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
    }
    std::exception_ptr failure() {
        std::lock_guard<std::mutex> lock(mutex_);
        return failure_;
    }
    void check() {
        auto error = failure();
        if (error) std::rethrow_exception(error);
    }
    void post(std::function<void()> command, bool allow_error = false) {
        OwnerWaitGuard wait_guard;
        std::unique_lock<std::mutex> lock(mutex_);
        space_.wait(lock, [&] { return stop_ || queue_.size() < capacity_; });
        TORCH_CHECK(!stop_, "cache owner has stopped");
        if (failure_ && !allow_error) std::rethrow_exception(failure_);
        queue_.push_back(std::move(command));
        queue_high_water_ = std::max<int64_t>(queue_high_water_, queue_.size());
        cv_.notify_one();
    }
    template <class F>
    auto sync(F fn, bool allow_error = false) -> decltype(fn()) {
        using Result = decltype(fn());
        auto task = std::make_shared<std::packaged_task<Result()>>(
            [this, fn = std::move(fn), allow_error]() mutable {
                if (!allow_error) check();
                return fn();
            });
        auto result = task->get_future();
        post([task] { (*task)(); }, allow_error);
        OwnerWaitGuard wait_guard;
        return result.get();
    }
    void initialize(at::Tensor anchor) {
        sync([&] {
            call("kvcache_manager_ops::init_kvcache", {anchor});
            initialized_ = true;
        });
    }
    void onboard(const std::shared_ptr<Request>& r) {
        // Called only by the owner. Publishing metadata precedes expensive waits.
        if (auto error = failure()) {
            std::lock_guard<std::mutex> lock(r->mutex);
            r->error = error;
            r->cv.notify_all();
            return;
        }
        if (!r->cpu_hits) {
            complete(r);
            return;
        }
        onboards_.push_back(r);
        has_work_ = true;
    }
    void submit_offload(std::shared_ptr<Request> r) {
        post([this, r = std::move(r)] {
            offloads_.push_back(r);
            has_work_ = true;
        });
    }
    void admission(at::Tensor ids) {
        // The preceding request has retired before prepare is called. Its event
        // must therefore be complete. Pin pending D2H slots before allocation.
        progress();
        TORCH_CHECK(offloads_.empty(),
                    "previous inference has not completed before new allocation");
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!call("kvcache_manager_ops::fixed_allocation_available", {ids})[0].toBool()) {
            ++admission_polls_;
            progress();
            TORCH_CHECK(std::chrono::steady_clock::now() < deadline,
                        "cache admission timed out waiting for pinned slots");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void flush() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        do {
            progress();
            if (!has_work_) return;
            TORCH_CHECK(std::chrono::steady_clock::now() < deadline, "cache owner flush timed out");
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        } while (true);
    }
    at::Tensor stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return ints({queue_high_water_, onboard_polls_, offload_launches_, offload_reaped_,
                     admission_polls_, inflight_offloads_, (int64_t)offloads_.size(),
                     (int64_t)onboards_.size()});
    }
    int64_t pending_offloads() {
        return inflight_offloads_ + static_cast<int64_t>(offloads_.size());
    }
    at::Tensor take_reaped() {
        auto result = at::empty({static_cast<int64_t>(completed_.size()), 3},
                                at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        auto* data = result.data_ptr<int64_t>();
        for (size_t i = 0; i < completed_.size(); ++i)
            for (size_t j = 0; j < 3; ++j) data[i * 3 + j] = completed_[i][j];
        completed_.clear();
        return result;
    }
    void wait_request_completion() {
        if (last_request_done_) last_request_done_->synchronize();
        last_request_done_.reset();
    }
    void record_request_completion() {
        auto event = std::make_shared<at::cuda::CUDAEvent>();
        event->record(at::cuda::getCurrentCUDAStream());
        post([this, event] { last_request_done_ = event; });
    }
    void shutdown(at::Tensor anchor) {
        wait_request_completion();
        if (initialized_) call("kvcache_manager_ops::shutdown_runtime", {anchor});
        initialized_ = false;
        has_work_ = false;
        onboards_.clear();
        offloads_.clear();
    }

   private:
    void complete(const std::shared_ptr<Request>& r) {
        std::lock_guard<std::mutex> lock(r->mutex);
        r->onboard_complete = true;
        r->cv.notify_all();
    }
    void fail(std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = error;
        }
        for (auto& r : onboards_) {
            std::lock_guard<std::mutex> lock(r->mutex);
            r->error = error;
            r->cv.notify_all();
        }
        has_work_ = false;
    }
    void progress() {
        if (!initialized_) return;
        for (auto it = offloads_.begin(); it != offloads_.end();) {
            auto r = *it;
            if (!r->done.query()) {
                ++it;
                continue;
            }
            if (r->compact_resident) {
                // The inference fast path owns no generic transfer metadata. Publication
                // is uncommon and asynchronous: build independent tensors now, on the
                // owner, rather than retaining/reusing the compact upload buffers.
                // progress() runs before a later allocation can evict these snapshots.
                r->found = call("kvcache_manager_ops::lookup", {r->ids, r->lengths, r->ids})[0]
                               .toTensorVector();
                for (int64_t i = 0; i < r->count; ++i)
                    TORCH_CHECK(r->found[1].data_ptr<int>()[i] == r->lengths.data_ptr<int64_t>()[i],
                                "resident publication lost its protected history");
                r->allocation = call("kvcache_manager_ops::allocate",
                                     {r->ids, r->lengths, r->found[1], r->found[5]})[0]
                                    .toTensorVector();
                r->compact_resident = false;
            }
            auto mapping = r->slot_mappings;
            if (mapping.empty()) mapping = {ints({-1})};
            auto tasks = call("kvcache_manager_ops::offload_launch",
                              {r->ids, r->lengths, r->found[1], r->found[5], r->found[2],
                               r->found[3], r->allocation[0], r->allocation[2],
                               c10::List<at::Tensor>(mapping), r->ids})[0]
                             .toTensor();
            for (int64_t i = 0; i < tasks.numel(); ++i) {
                auto task = tasks.data_ptr<int64_t>()[i];
                if (task >= 0) {
                    ++inflight_offloads_;
                    if (r->completion_ids.defined())
                        offload_ids_[task] = r->completion_ids.data_ptr<int64_t>()[i];
                } else if (r->completion_ids.defined()) {
                    completed_.push_back({r->ids.data_ptr<int64_t>()[i],
                                          r->completion_ids.data_ptr<int64_t>()[i], 0});
                }
            }
            ++offload_launches_;
            it = offloads_.erase(it);
        }
        for (auto it = onboards_.begin(); it != onboards_.end();) {
            auto r = *it;
            if (!r->onboard_launched) {
                auto launched = call("kvcache_manager_ops::onboard_launch",
                                     {r->ids, r->lengths, c10::List<at::Tensor>(r->found),
                                      r->allocation[0], r->allocation[2]});
                r->tasks = launched[2].toTensor();
                r->onboard_launched = true;
            }
            ++onboard_polls_;
            if (call("kvcache_manager_ops::onboard_poll", {r->tasks})[0].toBool()) {
                complete(r);
                it = onboards_.erase(it);
            } else
                ++it;
        }
        if (inflight_offloads_) {
            auto completed =
                call("kvcache_manager_ops::offload_reap_completed", {ints({})})[0].toTensor();
            for (int64_t i = 0; i < completed.size(0); ++i) {
                --inflight_offloads_;
                ++offload_reaped_;
                auto* row = completed.data_ptr<int64_t>() + 3 * i;
                auto found = offload_ids_.find(row[1]);
                if (found != offload_ids_.end()) {
                    completed_.push_back({row[0], found->second, row[2]});
                    offload_ids_.erase(found);
                }
                TORCH_CHECK(row[2] == 0, "FlexKV offload failed, task=", row[1]);
            }
        }
        has_work_ = !offloads_.empty() || !onboards_.empty() || inflight_offloads_;
    }
    std::shared_ptr<at::cuda::CUDAEvent> last_request_done_;
    std::unordered_map<int64_t, int64_t> offload_ids_;
    std::vector<std::array<int64_t, 3>> completed_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_, space_;
    std::deque<std::function<void()>> queue_;
    std::thread thread_;
    std::exception_ptr failure_;
    bool stop_ = false, initialized_ = false, has_work_ = false;
    std::vector<std::shared_ptr<Request>> onboards_, offloads_;
    int64_t queue_high_water_ = 0, onboard_polls_ = 0, offload_launches_ = 0, offload_reaped_ = 0,
            admission_polls_ = 0, inflight_offloads_ = 0;
};

CacheOwner::CacheOwner(int device, size_t capacity)
    : impl_(std::make_unique<Impl>(device, capacity)) {}
CacheOwner::~CacheOwner() = default;
void CacheOwner::post(std::function<void()> fn, bool allow) { impl_->post(std::move(fn), allow); }
void CacheOwner::initialize(at::Tensor anchor) { impl_->initialize(anchor); }
void CacheOwner::check() { impl_->check(); }
void CacheOwner::onboard(const std::shared_ptr<OwnerTransferState>& r) { impl_->onboard(r); }
void CacheOwner::submit_offload(std::shared_ptr<OwnerTransferState> r) {
    impl_->submit_offload(std::move(r));
}
void CacheOwner::admission(at::Tensor ids) { impl_->admission(ids); }
void CacheOwner::flush() { impl_->flush(); }
at::Tensor CacheOwner::stats() { return impl_->stats(); }
void CacheOwner::shutdown(at::Tensor anchor) { impl_->shutdown(anchor); }
int64_t CacheOwner::pending_offloads() { return impl_->pending_offloads(); }
at::Tensor CacheOwner::take_reaped() { return impl_->take_reaped(); }
void CacheOwner::wait_request_completion() { impl_->wait_request_completion(); }
void CacheOwner::record_request_completion() { impl_->record_request_completion(); }
}  // namespace kvcache_manager
