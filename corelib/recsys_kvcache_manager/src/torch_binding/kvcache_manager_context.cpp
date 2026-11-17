#include "kvcache_manager_context.h"
#include "export_kvcache_runtime.h"
#include "owner_kvcache_runtime.h"
#include "cache_owner.h"
#include "kvcache_runtime_config.h"

#include <ATen/ATen.h>
#include <iostream>
#include <thread>
#include <utility>

#ifndef KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD
#define KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD 0
#endif

#ifndef KVCACHE_MANAGER_CONTEXT_DEBUG
#define KVCACHE_MANAGER_CONTEXT_DEBUG 0
#endif

namespace kvcache_manager {

namespace {

void log_kvcache_context_message(const std::string& message) {
#if KVCACHE_MANAGER_CONTEXT_DEBUG
    std::cout << "[KVCACHE][context] " << message << std::endl;
#else
    (void)message;
#endif
}

} // namespace

#if KVCACHE_MANAGER_ENABLE_SCOPED_CONTEXT_GUARD
namespace {

// Optional helper: keep this local to the .cpp so it can be removed freely.
class ScopedKVCacheRuntimeContextGuard final {
public:
    explicit ScopedKVCacheRuntimeContextGuard(std::shared_ptr<IKVCacheRuntime> manager)
        : previous_(KVCacheRuntimeContext::instance().has_manager()
                        ? KVCacheRuntimeContext::instance().get_manager()
                        : nullptr) {
        KVCacheRuntimeContext::instance().set_manager(std::move(manager));
    }

    ~ScopedKVCacheRuntimeContextGuard() {
        if (restore_previous_) {
            KVCacheRuntimeContext::instance().set_manager(std::move(previous_));
        }
    }

    ScopedKVCacheRuntimeContextGuard(const ScopedKVCacheRuntimeContextGuard&) = delete;
    ScopedKVCacheRuntimeContextGuard& operator=(const ScopedKVCacheRuntimeContextGuard&) = delete;

    ScopedKVCacheRuntimeContextGuard(ScopedKVCacheRuntimeContextGuard&&) = delete;
    ScopedKVCacheRuntimeContextGuard& operator=(ScopedKVCacheRuntimeContextGuard&&) = delete;

    void dismiss() {
        restore_previous_ = false;
    }

private:
    std::shared_ptr<IKVCacheRuntime> previous_;
    bool restore_previous_ = true;
};

} // namespace
#endif

thread_local std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::kvcache_manager_ = nullptr;

KVCacheRuntimeContext& KVCacheRuntimeContext::instance() {
    static KVCacheRuntimeContext context;
    if (!context.has_manager()) {
        log_kvcache_context_message(
            "thread=" + c10::str(std::this_thread::get_id())
            + " creating ExportKVCacheRuntime"
        );
        auto config=load_yaml_config(find_config_path());
        bool owner_enabled=config.count("owner_loop") && get_config_int(config,"owner_loop")!=0;
        if(owner_enabled && !cache_owner_worker_thread())
            context.set_manager(make_owner_kvcache_runtime());
        else
            context.set_manager(std::make_shared<ExportKVCacheRuntime>());
        log_kvcache_context_message(
            "thread=" + c10::str(std::this_thread::get_id())
            + " ExportKVCacheRuntime ready"
        );
    }
    return context;
}

void KVCacheRuntimeContext::set_manager(std::shared_ptr<IKVCacheRuntime> manager) {
    log_kvcache_context_message(
        "thread=" + c10::str(std::this_thread::get_id())
        + " set_manager manager=" + c10::str(manager.get())
    );
    kvcache_manager_ = std::move(manager);
}

bool KVCacheRuntimeContext::has_manager() const {
    return static_cast<bool>(kvcache_manager_);
}

std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::manager() const {
    TORCH_CHECK(
        kvcache_manager_ != nullptr,
        "KVCacheRuntimeContext is empty for current thread. "
        "Call set_manager() before invoking kvcache_manager_ops.");
    return kvcache_manager_;
}

std::shared_ptr<IKVCacheRuntime> KVCacheRuntimeContext::get_manager() {
    return manager();
}

void KVCacheRuntimeContext::clear_manager() {
    log_kvcache_context_message(
        "thread=" + c10::str(std::this_thread::get_id())
        + " clear_manager manager=" + c10::str(kvcache_manager_.get())
    );
    shutdown_current();
}

void KVCacheRuntimeContext::shutdown_current() {
    if (!kvcache_manager_) return;
    // Keep the runtime and its GPU storage alive if draining fails or times
    // out. The caller can retry shutdown after resolving the server failure.
    kvcache_manager_->shutdown_runtime();
    kvcache_manager_.reset();
}

} // namespace kvcache_manager

