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

namespace recsys::nve_loader {
namespace {

constexpr const char* kNve2605Plugin =
    "/opt/nve/26.05/replay/librecsys_nve_loader.so";
constexpr const char* kNve2607Plugin =
    "/opt/nve/26.07/replay/librecsys_nve_loader.so";

[[noreturn]] void metadata_error(const std::string& detail) {
  throw std::runtime_error("NVE artifact metadata error: " + detail);
}

std::string classify_metadata(const nlohmann::json& metadata) {
  if (metadata.is_array()) {
    if (metadata.empty()) {
      metadata_error("legacy layer list is empty");
    }
    for (const auto& layer : metadata) {
      if (!layer.is_object() || !layer.contains("cache_type") ||
          layer.contains("layer_type")) {
        metadata_error(
            "legacy layers must contain cache_type and must not contain layer_type");
      }
    }
    return "26.05";
  }

  if (metadata.is_object()) {
    auto version = metadata.find("version");
    if (version == metadata.end() || !version->is_number_integer() ||
        version->get<int>() != 2) {
      metadata_error("unsupported schema version");
    }
    auto resources_it = metadata.find("resources");
    if (resources_it == metadata.end() || !resources_it->is_object()) {
      metadata_error("schema v2 resources must be an object");
    }
    auto layers_it = metadata.find("layers");
    if (layers_it == metadata.end() || !layers_it->is_array() ||
        layers_it->empty()) {
      metadata_error("schema v2 layers must be a non-empty list");
    }
    for (const auto& layer : *layers_it) {
      if (!layer.is_object() || !layer.contains("layer_type") ||
          layer.contains("cache_type")) {
        metadata_error(
            "schema v2 layers must contain layer_type and must not contain cache_type");
      }
    }
    return "26.07";
  }

  metadata_error("expected a legacy array or schema-v2 object");
}

std::string artifact_generation(const std::string& package_dir) {
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

std::string selected_generation() {
  const char* value = std::getenv("NVE_VERSION");
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(
        "NVE_VERSION must be set to exactly 26.05 or 26.07");
  }
  const std::string selected(value);
  if (selected != "26.05" && selected != "26.07") {
    throw std::runtime_error(
        "Unsupported NVE_VERSION=" + selected +
        "; expected exactly 26.05 or 26.07");
  }
  return selected;
}

const std::string& process_selected_generation() {
  // Function-local static initialization is thread-safe and deliberately
  // freezes NVE_VERSION at the first replay-manager construction.
  static const std::string selected = selected_generation();
  return selected;
}

const char* plugin_path_for(const std::string& generation) {
  return generation == "26.05" ? kNve2605Plugin : kNve2607Plugin;
}

RecsysNveInitPhase expected_phase_for(const std::string& generation) {
  return generation == "26.05" ? RECSYS_NVE_INIT_BEFORE_AOTI
                               : RECSYS_NVE_INIT_AFTER_AOTI;
}

std::string dl_error_or_unknown() {
  const char* error = dlerror();
  return error == nullptr ? "unknown dynamic-loader error" : error;
}

}  // namespace

struct NveLoaderPlugin::Impl {
  explicit Impl(std::string package) : package_dir(std::move(package)) {
    if (package_dir.empty()) {
      throw std::runtime_error("NVE package directory must not be empty");
    }

    selected_version = process_selected_generation();
    const std::string required_version = artifact_generation(package_dir);
    if (selected_version != required_version) {
      throw std::runtime_error(
          "NVE version mismatch: selected runtime " + selected_version +
          ", artifact requires " + required_version);
    }

    plugin_path = plugin_path_for(selected_version);
    dlerror();
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
    flags |= RTLD_NODELETE;
#endif
    handle = dlopen(plugin_path.c_str(), flags);
    if (handle == nullptr) {
      throw std::runtime_error(
          "Failed to load NVE " + selected_version + " plugin " + plugin_path +
          ": " + dl_error_or_unknown());
    }

    dlerror();
    auto* symbol = dlsym(handle, "recsys_nve_loader_get_api_v1");
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || symbol == nullptr) {
      throw std::runtime_error(
          "NVE loader plugin is missing recsys_nve_loader_get_api_v1: " +
          std::string(symbol_error == nullptr ? "symbol not found" : symbol_error));
    }
    auto getter = reinterpret_cast<RecsysNveLoaderGetApiV1>(symbol);
    api = getter();
    validate_api();
  }

  ~Impl() { destroy_state(); }

  void validate_api() const {
    if (api == nullptr) {
      throw std::runtime_error("NVE loader plugin returned a null API table");
    }
    if (api->abi_version != RECSYS_NVE_LOADER_ABI_VERSION) {
      throw std::runtime_error("NVE loader plugin ABI version mismatch");
    }
    if (api->struct_size < sizeof(RecsysNveLoaderApiV1)) {
      throw std::runtime_error("NVE loader plugin API table is truncated");
    }
    if (api->nve_version == nullptr || selected_version != api->nve_version) {
      throw std::runtime_error("NVE loader plugin claimed the wrong generation");
    }
    if (api->init_phase != expected_phase_for(selected_version)) {
      throw std::runtime_error("NVE loader plugin claimed the wrong init phase");
    }
    if (api->prepare_native_ops == nullptr || api->create_state == nullptr ||
        api->destroy_state == nullptr) {
      throw std::runtime_error("NVE loader plugin API table is incomplete");
    }
  }

  void prepare_native_ops() {
    if (prepared) {
      return;
    }
    std::array<char, 1024> error{};
    if (api->prepare_native_ops(error.data(), error.size()) != 0) {
      throw std::runtime_error(
          error[0] == '\0' ? "NVE native-op preparation failed" : error.data());
    }
    prepared = true;
  }

  void create_state(void* loader, int device_index) {
    if (!prepared) {
      throw std::runtime_error(
          "NVE native operators must be prepared before state creation");
    }
    if (state != nullptr) {
      throw std::runtime_error("NVE loader state was already created");
    }
    const bool wants_loader = api->init_phase == RECSYS_NVE_INIT_AFTER_AOTI;
    if (wants_loader != (loader != nullptr)) {
      throw std::runtime_error(
          wants_loader
              ? "NVE 26.07 state requires an AOTI loader"
              : "NVE 26.05 state must be created before the AOTI loader");
    }

    std::array<char, 1024> error{};
    if (api->create_state(
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

  void destroy_state() noexcept {
    if (state != nullptr && api != nullptr && api->destroy_state != nullptr) {
      api->destroy_state(state);
      state = nullptr;
    }
  }

  std::string package_dir;
  std::string selected_version;
  std::string plugin_path;
  void* handle = nullptr;  // Intentionally retained; never dlclose NVE operators.
  const RecsysNveLoaderApiV1* api = nullptr;
  void* state = nullptr;
  bool prepared = false;
};

NveLoaderPlugin::NveLoaderPlugin(std::string package_dir)
    : impl_(std::make_unique<Impl>(std::move(package_dir))) {}

NveLoaderPlugin::~NveLoaderPlugin() = default;
NveLoaderPlugin::NveLoaderPlugin(NveLoaderPlugin&&) noexcept = default;
NveLoaderPlugin& NveLoaderPlugin::operator=(NveLoaderPlugin&&) noexcept = default;

const std::string& NveLoaderPlugin::selected_version() const noexcept {
  return impl_->selected_version;
}

RecsysNveInitPhase NveLoaderPlugin::init_phase() const noexcept {
  return impl_->api->init_phase;
}

void NveLoaderPlugin::prepare_native_ops() {
  impl_->prepare_native_ops();
}

void NveLoaderPlugin::create_state(
    void* aoti_loader_or_null, int device_index) {
  impl_->create_state(aoti_loader_or_null, device_index);
}

void NveLoaderPlugin::destroy_state() noexcept {
  if (impl_ != nullptr) {
    impl_->destroy_state();
  }
}

}  // namespace recsys::nve_loader
