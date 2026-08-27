/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#if !defined(RECSYS_NVE_INSTALL_ROOT)
#error "CMake must provide the versioned NVE install root"
#endif

namespace recsys::nve_loader {
namespace {

constexpr const char* kNveInstallRoot = RECSYS_NVE_INSTALL_ROOT;

std::string dl_error_or_unknown() {
  const char* error = dlerror();
  return error == nullptr ? "unknown dynamic-loader error" : error;
}

void* required_symbol(void* handle, const char* name) {
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(
        "NVE loader plugin is missing " + std::string(name) + ": " +
        (error == nullptr ? "symbol not found" : error));
  }
  return symbol;
}

}  // namespace

NveLoaderPlugin::NveLoaderPlugin(std::string package_dir)
    : package_dir_(std::move(package_dir)) {
  if (package_dir_.empty()) {
    throw std::runtime_error("NVE package directory must not be empty");
  }

  const char* version = std::getenv("NVE_VERSION");
  if (version == nullptr || version[0] == '\0') {
    throw std::runtime_error(
        "NVE_VERSION must be set to exactly 26.05, 26.06, or 26.07");
  }
  selected_version_ = version;
  if (selected_version_ == "26.05") {
    uses_api_v2_ = false;
  } else if (selected_version_ == "26.06" ||
             selected_version_ == "26.07") {
    uses_api_v2_ = true;
  } else {
    throw std::runtime_error(
        "Unsupported NVE_VERSION=" + selected_version_ +
        "; expected exactly 26.05, 26.06, or 26.07");
  }

  const auto plugin_path = std::filesystem::path(kNveInstallRoot) /
      selected_version_ / "replay/librecsys_nve_loader.so";
  dlerror();
  void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(
        "Failed to load NVE " + selected_version_ + " plugin " +
        plugin_path.string() + ": " + dl_error_or_unknown());
  }
  // Do not dlclose: PyTorch retains operators registered by the plugin.

  create_state_fn_ = reinterpret_cast<CreateStateFn>(
      required_symbol(handle, "recsys_nve_loader_create_state"));
  destroy_state_fn_ = reinterpret_cast<DestroyStateFn>(
      required_symbol(handle, "recsys_nve_loader_destroy_state"));
}

NveLoaderPlugin::~NveLoaderPlugin() {
  if (state_ != nullptr) {
    destroy_state_fn_(state_);
  }
}

const std::string& NveLoaderPlugin::selected_version() const noexcept {
  return selected_version_;
}

bool NveLoaderPlugin::requires_aoti_loader() const noexcept {
  return uses_api_v2_;
}

void NveLoaderPlugin::create_state(
    void* aoti_loader_or_null, int device_index) {
  std::array<char, 1024> error{};
  state_ = create_state_fn_(
      package_dir_.c_str(),
      aoti_loader_or_null,
      device_index,
      error.data(),
      error.size());
  if (state_ == nullptr) {
    throw std::runtime_error(
        error[0] == '\0' ? "NVE loader state creation failed" : error.data());
  }
}

}  // namespace recsys::nve_loader
