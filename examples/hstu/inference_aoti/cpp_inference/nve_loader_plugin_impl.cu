/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <cstdio>
#include <exception>
#include <memory>

#if !defined(RECSYS_NVE_GENERATION)
#error "RECSYS_NVE_GENERATION must be 2605, 2606, or 2607"
#elif RECSYS_NVE_GENERATION != 2605 && RECSYS_NVE_GENERATION != 2606 && \
    RECSYS_NVE_GENERATION != 2607
#error "Unsupported RECSYS_NVE_GENERATION"
#endif

#include "python/pynve/torch_bindings/nve_loader.hpp"

namespace {

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

#if RECSYS_NVE_GENERATION == 2605
struct PluginState {
  std::unique_ptr<nve::LayerDirectory> layers;
};
#else
struct PluginState {
  std::shared_ptr<nve::ResourceDirectory> resources;
  std::unique_ptr<nve::LayerDirectory> layers;
};
#endif

}  // namespace

extern "C" RECSYS_NVE_EXPORT int recsys_nve_loader_create_state(
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
      set_error(
          error,
          error_size,
          "NVE state requires an AOTI loader");
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

extern "C" RECSYS_NVE_EXPORT void recsys_nve_loader_destroy_state(
    void* state) noexcept {
  delete static_cast<PluginState*>(state);
}
