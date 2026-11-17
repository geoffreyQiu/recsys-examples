// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <dlfcn.h>

namespace kvcache_manager {
namespace detail {
// Resolve optional CPython symbols without linking Python/libtorch_python.
// Native serving processes have no such symbols and take the no-op path.
struct PythonWaitHooks {
    using Check = int (*)();
    using Save = void* (*)();
    using Restore = void (*)(void*);
    Check initialized = reinterpret_cast<Check>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
    Check gil_held = reinterpret_cast<Check>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    Check finalizing = reinterpret_cast<Check>(dlsym(RTLD_DEFAULT, "Py_IsFinalizing"));
    Save save = reinterpret_cast<Save>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    Restore restore = reinterpret_cast<Restore>(dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));
    PythonWaitHooks() {
        if (!finalizing)
            finalizing = reinterpret_cast<Check>(dlsym(RTLD_DEFAULT, "_Py_IsFinalizing"));
    }
};
inline const PythonWaitHooks& python_wait_hooks() {
    static const PythonWaitHooks hooks;
    return hooks;
}
}  // namespace detail

// AOTI's Python runner can retain the GIL inside native operators. Tensor
// handle copies/releases on an owner worker can acquire that same GIL through
// PyTorch's PyObjectSlot refcount hooks. Release an already-held GIL while
// waiting for the worker; do nothing in C++ serving or GIL-free callers.
// Declare BEFORE a wait mutex, so mutex unlock precedes GIL restoration.
class OwnerWaitGuard {
public:
    OwnerWaitGuard() {
        const auto& hooks = detail::python_wait_hooks();
        if (hooks.initialized && hooks.gil_held && hooks.save && hooks.restore &&
            hooks.initialized() && (!hooks.finalizing || !hooks.finalizing()) && hooks.gil_held())
            python_thread_state_ = hooks.save();
    }
    ~OwnerWaitGuard() {
        if (python_thread_state_) detail::python_wait_hooks().restore(python_thread_state_);
    }
    OwnerWaitGuard(const OwnerWaitGuard&) = delete;
    OwnerWaitGuard& operator=(const OwnerWaitGuard&) = delete;
private:
    void* python_thread_state_ = nullptr;
};
}  // namespace kvcache_manager
