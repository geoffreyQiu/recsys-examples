/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <cstdio>
#include <exception>
#include <memory>

#if !defined(RECSYS_NVE_LOADER_API)
#error "RECSYS_NVE_LOADER_API must be 1 or 2"
#elif RECSYS_NVE_LOADER_API != 1 && RECSYS_NVE_LOADER_API != 2
#error "Unsupported RECSYS_NVE_LOADER_API"
#endif

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

void set_error(
    char* error, std::size_t error_size, const char* message) noexcept {
  if (error != nullptr && error_size > 0) {
    std::snprintf(error, error_size, "%s", message);
  }
}

struct PluginState {
#if RECSYS_NVE_LOADER_API == 2
  std::shared_ptr<nve::ResourceDirectory> resources;
#endif
  std::unique_ptr<nve::LayerDirectory> layers;
};

}  // namespace

extern "C" RECSYS_NVE_EXPORT void* recsys_nve_loader_create_state(
    const char* package_dir,
    void* aoti_loader_or_null,
    int device_index,
    char* error,
    std::size_t error_size) noexcept {
  if (error != nullptr && error_size > 0) {
    error[0] = '\0';
  }
  try {
#if RECSYS_NVE_LOADER_API == 1
    if (aoti_loader_or_null != nullptr) {
      set_error(error, error_size, "NVE 26.05 state must be created before AOTI");
      return nullptr;
    }
#else
    if (aoti_loader_or_null == nullptr) {
      set_error(error, error_size, "NVE state requires an AOTI loader");
      return nullptr;
    }
#endif
    auto state = std::make_unique<PluginState>();
#if RECSYS_NVE_LOADER_API == 1
    state->layers =
        std::make_unique<nve::LayerDirectory>(package_dir, device_index);
#else
    auto* loader = static_cast<torch::inductor::AOTIModelPackageLoader*>(
        aoti_loader_or_null);
    state->resources = std::make_shared<nve::ResourceDirectory>();
    state->layers = std::make_unique<nve::LayerDirectory>(
        package_dir, *loader, device_index, state->resources);
#endif
    return state.release();
  } catch (const std::exception& exception) {
    set_error(error, error_size, exception.what());
  } catch (...) {
    set_error(error, error_size, "unknown exception during NVE state creation");
  }
  return nullptr;
}

extern "C" RECSYS_NVE_EXPORT void recsys_nve_loader_destroy_state(
    void* state) noexcept {
  delete static_cast<PluginState*>(state);
}
