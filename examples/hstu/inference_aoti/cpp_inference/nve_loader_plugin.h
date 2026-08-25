/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#define RECSYS_NVE_LOADER_ABI_VERSION UINT32_C(1)

typedef uint32_t RecsysNveInitPhase;
#define RECSYS_NVE_INIT_BEFORE_AOTI UINT32_C(1)
#define RECSYS_NVE_INIT_AFTER_AOTI UINT32_C(2)

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

typedef struct RecsysNveLoaderApiV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* nve_version;
  RecsysNveInitPhase init_phase;
  int (*prepare_native_ops)(char* error, size_t error_size);
  int (*create_state)(
      const char* package_dir,
      void* aoti_loader_or_null,
      int device_index,
      void** state,
      char* error,
      size_t error_size);
  void (*destroy_state)(void* state) RECSYS_NVE_NOEXCEPT;
} RecsysNveLoaderApiV1;

typedef const RecsysNveLoaderApiV1* (*RecsysNveLoaderGetApiV1)(void)
    RECSYS_NVE_NOEXCEPT;

RECSYS_NVE_EXPORT const RecsysNveLoaderApiV1* recsys_nve_loader_get_api_v1(void)
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
  NveLoaderPlugin(NveLoaderPlugin&&) noexcept;
  NveLoaderPlugin& operator=(NveLoaderPlugin&&) noexcept;

  const std::string& selected_version() const noexcept;
  RecsysNveInitPhase init_phase() const noexcept;
  void prepare_native_ops();
  void create_state(void* aoti_loader_or_null, int device_index);
  void destroy_state() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace recsys::nve_loader
#endif
