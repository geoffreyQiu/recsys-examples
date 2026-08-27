/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
#define RECSYS_NVE_NOEXCEPT noexcept
extern "C" {
#else
#define RECSYS_NVE_NOEXCEPT
#endif

#if defined(__GNUC__) || defined(__clang__)
#define RECSYS_NVE_EXPORT __attribute__((visibility("default")))
#else
#define RECSYS_NVE_EXPORT
#endif

RECSYS_NVE_EXPORT int recsys_nve_loader_create_state(
    const char* package_dir,
    void* aoti_loader_or_null,
    int device_index,
    void** state,
    char* error,
    size_t error_size);
RECSYS_NVE_EXPORT void recsys_nve_loader_destroy_state(void* state)
    RECSYS_NVE_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"

#include <memory>
#include <string>

namespace recsys::nve_loader {

class NveLoaderPlugin {
 public:
  explicit NveLoaderPlugin(std::string package_dir);
  ~NveLoaderPlugin();

  NveLoaderPlugin(const NveLoaderPlugin&) = delete;
  NveLoaderPlugin& operator=(const NveLoaderPlugin&) = delete;

  const std::string& selected_version() const noexcept;
  bool requires_aoti_loader() const noexcept;
  void create_state(void* aoti_loader_or_null, int device_index);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace recsys::nve_loader
#endif
