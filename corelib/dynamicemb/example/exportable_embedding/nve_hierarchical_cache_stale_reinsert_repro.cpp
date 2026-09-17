// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reproduce stale tier-1 reinsertion, then replace each per-tier erase with
// the corresponding per-tier update. Run as: <program> erase|update.

#include <cuda_runtime.h>
#include <nve_c_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CHECK_NVE(call)                                                        \
  do {                                                                         \
    const nve_status_t status = (call);                                         \
    if (status != NVE_SUCCESS) {                                                \
      const char* message = nullptr;                                            \
      nve_get_last_error(&message);                                             \
      std::fprintf(stderr, "NVE error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   message ? message : "unknown");                            \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

namespace {

constexpr int64_t kRowSize = sizeof(float);
constexpr int64_t kGpuTier = 0;
constexpr int64_t kHostTier = 1;
constexpr int64_t kRedisTier = 2;

enum class Operation { kErase, kUpdate };

void wait_for(nve_context_t context) {
  CHECK_NVE(nve_context_wait(context));
  CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
T* copy_to_device(const std::vector<T>& values) {
  T* device_values = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_values),
                        values.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(device_values, values.data(),
                        values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return device_values;
}

void insert(nve_layer_t layer, nve_context_t context,
            const std::vector<int64_t>& keys,
            const std::vector<float>& values, int64_t table_id) {
  auto* device_keys = copy_to_device(keys);
  auto* device_values = copy_to_device(values);
  CHECK_NVE(nve_layer_insert(layer, context, static_cast<int64_t>(keys.size()),
                             device_keys, kRowSize, kRowSize, device_values,
                             table_id));
  wait_for(context);
  CHECK_CUDA(cudaFree(device_values));
  CHECK_CUDA(cudaFree(device_keys));
}

void update(nve_layer_t layer, nve_context_t context,
            const std::vector<int64_t>& keys,
            const std::vector<float>& values, int64_t table_id) {
  auto* device_keys = copy_to_device(keys);
  auto* device_values = copy_to_device(values);
  CHECK_NVE(nve_layer_update(layer, context, static_cast<int64_t>(keys.size()),
                             device_keys, kRowSize, kRowSize, device_values,
                             table_id));
  wait_for(context);
  CHECK_CUDA(cudaFree(device_values));
  CHECK_CUDA(cudaFree(device_keys));
}

void erase(nve_layer_t layer, nve_context_t context,
           const std::vector<int64_t>& keys, int64_t table_id) {
  auto* device_keys = copy_to_device(keys);
  CHECK_NVE(nve_layer_erase(layer, context, static_cast<int64_t>(keys.size()),
                            device_keys, table_id));
  wait_for(context);
  CHECK_CUDA(cudaFree(device_keys));
}

std::vector<float> lookup(nve_layer_t layer, nve_context_t context,
                          const std::vector<int64_t>& keys) {
  auto* device_keys = copy_to_device(keys);
  float* device_output = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output),
                        keys.size() * kRowSize));
  CHECK_NVE(nve_layer_lookup(layer, context, static_cast<int64_t>(keys.size()),
                             device_keys, device_output, kRowSize, nullptr,
                             nullptr));
  wait_for(context);

  std::vector<float> output(keys.size());
  CHECK_CUDA(cudaMemcpy(output.data(), device_output,
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(device_output));
  CHECK_CUDA(cudaFree(device_keys));
  return output;
}

float find(nve_table_t table, nve_context_t context, int64_t key) {
  uint64_t hit_mask = 0;
  float value = 0.0f;
  CHECK_NVE(nve_table_find(table, context, 1, &key, &hit_mask, kRowSize,
                           &value, nullptr));
  CHECK_NVE(nve_context_wait(context));
  return (hit_mask & 1U) ? value : 0.0f;
}

void change_tier(Operation operation, nve_layer_t layer,
                 nve_context_t context, const std::vector<int64_t>& key,
                 const std::vector<float>& new_value, int64_t table_id) {
  if (operation == Operation::kErase) {
    erase(layer, context, key, table_id);
  } else {
    update(layer, context, key, new_value, table_id);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string(argv[1]) != "erase" &&
                    std::string(argv[1]) != "update")) {
    std::fprintf(stderr, "usage: %s erase|update\n", argv[0]);
    return 2;
  }
  const Operation operation = std::string(argv[1]) == "erase"
                                  ? Operation::kErase
                                  : Operation::kUpdate;

  CHECK_CUDA(cudaSetDevice(0));
  CHECK_NVE(nve_load_host_table_plugin("libnve-plugin-nvhm.so"));
  CHECK_NVE(nve_load_host_table_plugin("libnve-plugin-redis.so"));

  nve_host_factory_t host_factory = nullptr;
  nve_host_factory_t redis_factory = nullptr;
  CHECK_NVE(nve_create_host_table_factory(
      &host_factory, R"({"implementation":"nvhm_map"})"));
  CHECK_NVE(nve_create_host_table_factory(
      &redis_factory,
      R"({"implementation":"redis_cluster","address":"127.0.0.1:6379","single_node":true})"));

  nve_table_t host_table = nullptr;
  nve_table_t redis_table = nullptr;
  CHECK_NVE(nve_host_factory_produce(
      host_factory, 1,
      R"({"mask_size":8,"key_size":8,"max_value_size":4,"value_dtype":"float32","num_partitions":1,"initial_capacity":1024,"value_alignment":32})",
      &host_table));
  CHECK_NVE(nve_host_factory_produce(
      redis_factory, 2,
      R"({"mask_size":8,"key_size":8,"max_value_size":4,"value_dtype":"float32","num_partitions":0,"string_namespace_id":992})",
      &redis_table));

  auto gpu_config = nve_gpu_table_config_default();
  gpu_config.device_id = 0;
  gpu_config.cache_size = 1 << 20;
  gpu_config.row_size_in_bytes = kRowSize;
  gpu_config.value_dtype = NVE_DTYPE_FLOAT32;
  nve_table_t gpu_table = nullptr;
  CHECK_NVE(
      nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_config, nullptr));

  const float zero = 0.0f;
  auto layer_config = nve_hierarchical_layer_config_default();
  layer_config.layer_name = "stale_reinsert_repro";
  layer_config.default_embedding = &zero;
  layer_config.default_embedding_size = kRowSize;
  nve_table_t tables[] = {gpu_table, host_table, redis_table};
  nve_layer_t layer = nullptr;
  CHECK_NVE(nve_hierarchical_layer_create(
      &layer, NVE_KEY_INT64, &layer_config, tables, 3, nullptr));

  nve_context_t layer_context = nullptr;
  nve_context_t host_context = nullptr;
  nve_context_t redis_context = nullptr;
  CHECK_NVE(nve_layer_create_execution_context(
      layer, &layer_context, nullptr, nullptr, nullptr, nullptr));
  CHECK_NVE(nve_table_create_execution_context(
      host_table, &host_context, nullptr, nullptr, nullptr, nullptr));
  CHECK_NVE(nve_table_create_execution_context(
      redis_table, &redis_context, nullptr, nullptr, nullptr, nullptr));
  CHECK_NVE(nve_table_clear(redis_table, redis_context));
  CHECK_NVE(nve_context_wait(redis_context));

  const std::vector<int64_t> updated_key{7};
  const std::vector<float> old_value{1.0f};
  const std::vector<float> new_value{2.0f};

  insert(layer, layer_context, {7, 8, 9}, {1.0f, 8.0f, 9.0f},
         kRedisTier);
  insert(layer, layer_context, updated_key, old_value, kHostTier);

  // Step 1: Resolve key 7 from tier 1 while key 8 reaches Redis. This seeds
  // the two-row host gather buffer used by the later lookup.
  const float initial = lookup(layer, layer_context, {7, 8})[0];
  std::printf("initial=%.1f\n", initial);

  // Step 2: Establish the original steady state in tier 0.
  insert(layer, layer_context, updated_key, old_value, kGpuTier);

  // Step 3: The Redis value changes to 2.0.
  update(layer, layer_context, updated_key, new_value, kRedisTier);

  // Step 4: Change only tier 1. The update mode keeps the row resident.
  change_tier(operation, layer, layer_context, updated_key, new_value,
              kHostTier);

  // Step 5: A non-stop inference lookup runs between the two tier changes.
  const float between_tiers = lookup(layer, layer_context, {7, 9})[0];
  std::printf("between_tiers=%.1f\n", between_tiers);

  // Step 6: Change only tier 0.
  change_tier(operation, layer, layer_context, updated_key, new_value,
              kGpuTier);

  // Step 7: Inspect tier 1 before another hierarchical lookup can refill it.
  const float tier_1_after_changes =
      find(host_table, host_context, updated_key[0]);
  std::printf("tier_1_after_changes=%.1f\n", tier_1_after_changes);

  // Step 8: Compare the hierarchical result with the value stored in Redis.
  const float after_tier_0_change =
      lookup(layer, layer_context, updated_key)[0];
  const float expected_from_redis =
      find(redis_table, redis_context, updated_key[0]);
  std::printf("after_tier_0_change=%.1f\n", after_tier_0_change);
  std::printf("expected_from_redis=%.1f\n", expected_from_redis);

  const bool verified =
      operation == Operation::kErase
          ? (after_tier_0_change == initial &&
             after_tier_0_change != expected_from_redis)
          : (tier_1_after_changes == expected_from_redis &&
             after_tier_0_change == expected_from_redis);
  if (operation == Operation::kErase) {
    std::printf("RESULT original_stale_reinsertion_reproduced=%s\n",
                verified ? "true" : "false");
  } else {
    std::printf("RESULT tier_specific_update_resolved=%s\n",
                verified ? "true" : "false");
  }

  wait_for(layer_context);
  CHECK_NVE(nve_context_destroy(redis_context));
  CHECK_NVE(nve_context_destroy(host_context));
  CHECK_NVE(nve_context_destroy(layer_context));
  CHECK_NVE(nve_layer_destroy(layer));
  CHECK_NVE(nve_table_destroy(redis_table));
  CHECK_NVE(nve_table_destroy(host_table));
  CHECK_NVE(nve_table_destroy(gpu_table));
  CHECK_NVE(nve_host_factory_destroy(redis_factory));
  CHECK_NVE(nve_host_factory_destroy(host_factory));
  return verified ? 0 : 1;
}
