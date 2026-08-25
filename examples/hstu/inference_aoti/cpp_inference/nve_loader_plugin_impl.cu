/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>

#if !defined(RECSYS_NVE_GENERATION)
#error "RECSYS_NVE_GENERATION must be 2605 or 2607"
#elif RECSYS_NVE_GENERATION != 2605 && RECSYS_NVE_GENERATION != 2607
#error "Unsupported RECSYS_NVE_GENERATION"
#endif

#if !defined(RECSYS_NVE_LIBRARY_DIR)
#error "RECSYS_NVE_LIBRARY_DIR must name the matching versioned NVE library directory"
#endif

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

std::atomic<bool> g_native_ops_prepared{false};
std::mutex g_prepare_mutex;
void* g_native_ops_handle = nullptr;

bool valid_error_buffer(char* error, size_t error_size) {
  return error != nullptr && error_size > 0;
}

void set_error(char* error, size_t error_size, const char* message) noexcept {
  if (!valid_error_buffer(error, error_size)) {
    return;
  }
  std::snprintf(error, error_size, "%s", message == nullptr ? "unknown error" : message);
  error[error_size - 1] = '\0';
}

void set_error(
    char* error, size_t error_size, const std::string& message) noexcept {
  set_error(error, error_size, message.c_str());
}

int prepare_native_ops(char* error, size_t error_size) {
  if (!valid_error_buffer(error, error_size)) {
    return 1;
  }
  error[0] = '\0';
  try {
    std::lock_guard<std::mutex> lock(g_prepare_mutex);
    if (g_native_ops_handle == nullptr) {
      const std::string path =
          std::string(RECSYS_NVE_LIBRARY_DIR) + "/libnve-torch-ops.so";
      dlerror();
      int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
      flags |= RTLD_NODELETE;
#endif
      g_native_ops_handle = dlopen(path.c_str(), flags);
      if (g_native_ops_handle == nullptr) {
        const char* dl_error = dlerror();
        set_error(
            error,
            error_size,
            "failed to load matching NVE native operators from " + path +
                ": " + (dl_error == nullptr ? "unknown error" : dl_error));
        return 1;
      }
    }
    g_native_ops_prepared.store(true, std::memory_order_release);
    return 0;
  } catch (const std::exception& exception) {
    set_error(error, error_size, exception.what());
  } catch (...) {
    set_error(error, error_size, "unknown exception during NVE native-op preparation");
  }
  return 1;
}

#if RECSYS_NVE_GENERATION == 2605
struct PluginState {
  std::unique_ptr<nve::LayerDirectory> layers;
};
constexpr const char* kNveVersion = "26.05";
constexpr RecsysNveInitPhase kInitPhase = RECSYS_NVE_INIT_BEFORE_AOTI;
#else
struct PluginState {
  std::shared_ptr<nve::ResourceDirectory> resources;
  std::unique_ptr<nve::LayerDirectory> layers;
};
constexpr const char* kNveVersion = "26.07";
constexpr RecsysNveInitPhase kInitPhase = RECSYS_NVE_INIT_AFTER_AOTI;
#endif

int create_state(
    const char* package_dir,
    void* aoti_loader_or_null,
    int device_index,
    void** state,
    char* error,
    size_t error_size) {
  if (!valid_error_buffer(error, error_size)) {
    return 1;
  }
  error[0] = '\0';
  if (state == nullptr) {
    set_error(error, error_size, "state output pointer is null");
    return 1;
  }
  *state = nullptr;
  if (package_dir == nullptr || package_dir[0] == '\0') {
    set_error(error, error_size, "package directory is empty");
    return 1;
  }
  if (device_index < 0) {
    set_error(error, error_size, "NVE replay requires a non-negative CUDA device index");
    return 1;
  }
  if (!g_native_ops_prepared.load(std::memory_order_acquire)) {
    set_error(error, error_size, "prepare_native_ops must run before create_state");
    return 1;
  }

  try {
    auto owned = std::make_unique<PluginState>();
#if RECSYS_NVE_GENERATION == 2605
    if (aoti_loader_or_null != nullptr) {
      set_error(error, error_size, "NVE 26.05 state must be created before AOTI");
      return 1;
    }
    owned->layers =
        std::make_unique<nve::LayerDirectory>(package_dir, device_index);
#else
    if (aoti_loader_or_null == nullptr) {
      set_error(error, error_size, "NVE 26.07 state requires an AOTI loader");
      return 1;
    }
    auto* loader = static_cast<torch::inductor::AOTIModelPackageLoader*>(
        aoti_loader_or_null);
    owned->resources = std::make_shared<nve::ResourceDirectory>();
    owned->layers = std::make_unique<nve::LayerDirectory>(
        package_dir, *loader, device_index, owned->resources);
#endif
    *state = owned.release();
    return 0;
  } catch (const std::exception& exception) {
    set_error(error, error_size, exception.what());
  } catch (...) {
    set_error(error, error_size, "unknown exception during NVE state creation");
  }
  return 1;
}

void destroy_state(void* state) noexcept {
  try {
    delete static_cast<PluginState*>(state);
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "NVE loader cleanup failed: %s\n", exception.what());
  } catch (...) {
    std::fprintf(stderr, "NVE loader cleanup failed with an unknown exception\n");
  }
}

const RecsysNveLoaderApiV1 kApi = {
    RECSYS_NVE_LOADER_ABI_VERSION,
    sizeof(RecsysNveLoaderApiV1),
    kNveVersion,
    kInitPhase,
    &prepare_native_ops,
    &create_state,
    &destroy_state,
};

}  // namespace

extern "C" RECSYS_NVE_EXPORT
const RecsysNveLoaderApiV1* recsys_nve_loader_get_api_v1(void) noexcept {
  return &kApi;
}
