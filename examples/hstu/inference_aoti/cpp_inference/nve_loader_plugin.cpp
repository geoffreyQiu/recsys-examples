/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nve_loader_plugin.h"

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#if !defined(RECSYS_NVE_2605_PLUGIN_PATH) || \
    !defined(RECSYS_NVE_2606_PLUGIN_PATH) || \
    !defined(RECSYS_NVE_2607_PLUGIN_PATH)
#error "CMake must provide all versioned NVE replay-plugin paths"
#endif

namespace recsys::nve_loader {
namespace {

constexpr const char* kNve2605Plugin = RECSYS_NVE_2605_PLUGIN_PATH;
constexpr const char* kNve2606Plugin = RECSYS_NVE_2606_PLUGIN_PATH;
constexpr const char* kNve2607Plugin = RECSYS_NVE_2607_PLUGIN_PATH;

enum class NveArtifactContract {
  kLegacyV1,
  kSchemaV2,
};

[[noreturn]] void metadata_error(const std::string& detail) {
  throw std::runtime_error("NVE artifact metadata error: " + detail);
}

const char* contract_name(NveArtifactContract contract) {
  return contract == NveArtifactContract::kLegacyV1 ? "legacy-v1"
                                                    : "schema-v2";
}

NveArtifactContract classify_metadata(const nlohmann::json& metadata) {
  if (metadata.is_array()) {
    return NveArtifactContract::kLegacyV1;
  }

  if (metadata.is_object()) {
    auto version = metadata.find("version");
    if (version != metadata.end() && version->is_number_integer() &&
        version->get<int>() == 2) {
      return NveArtifactContract::kSchemaV2;
    }
    metadata_error("unsupported schema version");
  }

  metadata_error("expected a legacy array or schema-v2 object");
}

NveArtifactContract artifact_contract(const std::string& package_dir) {
  const auto metadata_path =
      std::filesystem::path(package_dir) / "metadata.json";
  std::ifstream input(metadata_path);
  if (!input.is_open()) {
    metadata_error("cannot open " + metadata_path.string());
  }
  try {
    nlohmann::json metadata;
    input >> metadata;
    return classify_metadata(metadata);
  } catch (const nlohmann::json::exception& error) {
    metadata_error("cannot parse " + metadata_path.string() + ": " + error.what());
  }
}

NveArtifactContract contract_for_version(const std::string& version) {
  if (version == "26.05") {
    return NveArtifactContract::kLegacyV1;
  }
  if (version == "26.06" || version == "26.07") {
    return NveArtifactContract::kSchemaV2;
  }
  throw std::runtime_error("Unsupported NVE version=" + version);
}

std::string selected_version_from_env() {
  const char* value = std::getenv("NVE_VERSION");
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(
        "NVE_VERSION must be set to exactly 26.05, 26.06, or 26.07");
  }
  const std::string selected(value);
  if (selected != "26.05" && selected != "26.06" &&
      selected != "26.07") {
    throw std::runtime_error(
        "Unsupported NVE_VERSION=" + selected +
        "; expected exactly 26.05, 26.06, or 26.07");
  }
  return selected;
}

const char* plugin_path_for(const std::string& version) {
  if (version == "26.05") {
    return kNve2605Plugin;
  }
  return version == "26.06" ? kNve2606Plugin : kNve2607Plugin;
}

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

using CreateStateFn = decltype(&recsys_nve_loader_create_state);
using DestroyStateFn = decltype(&recsys_nve_loader_destroy_state);

}  // namespace

struct NveLoaderPlugin::Impl {
  explicit Impl(std::string package) : package_dir(std::move(package)) {
    if (package_dir.empty()) {
      throw std::runtime_error("NVE package directory must not be empty");
    }

    selected_version = selected_version_from_env();
    const auto runtime_contract = contract_for_version(selected_version);
    const auto required_contract = artifact_contract(package_dir);
    if (runtime_contract != required_contract) {
      throw std::runtime_error(
          "NVE contract mismatch: selected runtime " + selected_version +
          " uses " + contract_name(runtime_contract) + ", artifact requires " +
          contract_name(required_contract));
    }

    const char* plugin_path = plugin_path_for(selected_version);
    dlerror();
    void* handle = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      throw std::runtime_error(
          "Failed to load NVE " + selected_version + " plugin " + plugin_path +
          ": " + dl_error_or_unknown());
    }
    // Do not dlclose: PyTorch retains operators registered by the plugin.

    create_state_fn = reinterpret_cast<CreateStateFn>(
        required_symbol(handle, "recsys_nve_loader_create_state"));
    destroy_state_fn = reinterpret_cast<DestroyStateFn>(
        required_symbol(handle, "recsys_nve_loader_destroy_state"));
  }

  ~Impl() {
    if (state != nullptr) {
      destroy_state_fn(state);
    }
  }

  void create_state(void* loader, int device_index) {
    std::array<char, 1024> error{};
    if (create_state_fn(
            package_dir.c_str(),
            loader,
            device_index,
            &state,
            error.data(),
            error.size()) != 0) {
      state = nullptr;
      throw std::runtime_error(
          error[0] == '\0' ? "NVE loader state creation failed" : error.data());
    }
    if (state == nullptr) {
      throw std::runtime_error("NVE loader plugin returned a null state");
    }
  }

  std::string package_dir;
  std::string selected_version;
  CreateStateFn create_state_fn = nullptr;
  DestroyStateFn destroy_state_fn = nullptr;
  void* state = nullptr;
};

NveLoaderPlugin::NveLoaderPlugin(std::string package_dir)
    : impl_(std::make_unique<Impl>(std::move(package_dir))) {}

NveLoaderPlugin::~NveLoaderPlugin() = default;

const std::string& NveLoaderPlugin::selected_version() const noexcept {
  return impl_->selected_version;
}

bool NveLoaderPlugin::requires_aoti_loader() const noexcept {
  return impl_->selected_version != "26.05";
}

void NveLoaderPlugin::create_state(
    void* aoti_loader_or_null, int device_index) {
  impl_->create_state(aoti_loader_or_null, device_index);
}

}  // namespace recsys::nve_loader
