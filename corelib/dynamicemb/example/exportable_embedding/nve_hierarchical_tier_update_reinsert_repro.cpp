// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Show that a lookup between a tier-1 update and a tier-0 update can overwrite
// the new tier-1 value through lookup-triggered NVHashMap auto-insertion.

#include <cuda_runtime.h>
#include <nve_c_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

void update(nve_layer_t layer, nve_context_t context, int64_t key,
            float value, int64_t table_id) {
  auto* device_key = copy_to_device(std::vector<int64_t>{key});
  auto* device_value = copy_to_device(std::vector<float>{value});
  CHECK_NVE(nve_layer_update(layer, context, 1, device_key, kRowSize,
                             kRowSize, device_value, table_id));
  wait_for(context);
  CHECK_CUDA(cudaFree(device_value));
  CHECK_CUDA(cudaFree(device_key));
}

void erase(nve_layer_t layer, nve_context_t context, int64_t key,
           int64_t table_id) {
  auto* device_key = copy_to_device(std::vector<int64_t>{key});
  CHECK_NVE(nve_layer_erase(layer, context, 1, device_key, table_id));
  wait_for(context);
  CHECK_CUDA(cudaFree(device_key));
}

void redis_insert(nve_table_t redis_table, nve_context_t redis_context,
                  int64_t key, float value) {
  CHECK_NVE(nve_table_insert(redis_table, redis_context, 1, &key, kRowSize,
                             kRowSize, &value));
  CHECK_NVE(nve_context_wait(redis_context));
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

}  // namespace

int main() {
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
      R"({"mask_size":8,"key_size":8,"max_value_size":4,"value_dtype":"float32","num_partitions":0,"string_namespace_id":993})",
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
  layer_config.layer_name = "tier_update_reinsert_repro";
  layer_config.min_insert_size_gpu = 1 << 16;
  layer_config.min_insert_size_host = 0;
  layer_config.default_embedding = &zero;
  layer_config.default_embedding_size = kRowSize;
  nve_table_t tables[] = {gpu_table, host_table, redis_table};
  nve_layer_t layer = nullptr;
  CHECK_NVE(nve_hierarchical_layer_create(
      &layer, NVE_KEY_INT64, &layer_config, tables, 3, nullptr));

  nve_context_t layer_context = nullptr;
  nve_context_t redis_context = nullptr;
  CHECK_NVE(nve_layer_create_execution_context(
      layer, &layer_context, nullptr, nullptr, nullptr, nullptr));
  CHECK_NVE(nve_table_create_execution_context(
      redis_table, &redis_context, nullptr, nullptr, nullptr, nullptr));

  constexpr int64_t key = 7;
  constexpr float old_value = 1.0f;
  constexpr float new_value = 2.0f;

  // Setup: Redis has the original row plus two filler rows. Tier 1 contains
  // the original test row, while tier 0 is initially empty.
  insert(layer, layer_context, {7, 8, 9}, {1.0f, 8.0f, 9.0f},
         kRedisTier);
  insert(layer, layer_context, {key}, {old_value}, kHostTier);

  // Step 1: Read the original value from tier 1 and seed the two-row host
  // gather buffer used by the later mixed-tier lookup.
  const float initial = lookup(layer, layer_context, {7, 8})[0];
  std::printf("initial=%.1f\n", initial);

  // Setup: establish the original value in tier 0.
  insert(layer, layer_context, {key}, {old_value}, kGpuTier);

  // Step 2: Represent the external Redis value change. This is an insert into
  // Redis, not another hierarchical-layer update or cache operation.
  redis_insert(redis_table, redis_context, key, new_value);

  // Step 3: The only tier-1 update: 1.0 -> 2.0 for the test key.
  update(layer, layer_context, key, new_value, kHostTier);

  // Step 4: A non-stop inference lookup runs before tier 0 is updated. It
  // returns tier 0's old 1.0 and triggers tier-1 auto-insertion for the batch.
  const float between_updates = lookup(layer, layer_context, {7, 9})[0];
  std::printf("between_updates=%.1f\n", between_updates);

  // Step 5: The only tier-0 update: 1.0 -> 2.0 for the test key.
  update(layer, layer_context, key, new_value, kGpuTier);

  // Step 6: Before either erase, a normal hierarchical lookup observes the
  // updated tier-0 value.
  const float after_both_updates = lookup(layer, layer_context, {key})[0];
  std::printf("after_both_updates=%.1f\n", after_both_updates);

  // Step 7: Erase tier 0 only. The next lookup exposes the value that the
  // intervening lookup reinserted into tier 1.
  erase(layer, layer_context, key, kGpuTier);
  const float exposed_tier_1 = lookup(layer, layer_context, {key})[0];
  std::printf("after_tier_0_erase=%.1f\n", exposed_tier_1);

  // Step 8: Erase tier 1 only. The next lookup reaches Redis and returns the
  // true updated value.
  erase(layer, layer_context, key, kHostTier);
  const float true_redis_value = lookup(layer, layer_context, {key})[0];
  std::printf("after_tier_1_erase=%.1f\n", true_redis_value);

  const bool reproduced = between_updates == old_value &&
                          after_both_updates == new_value &&
                          exposed_tier_1 != true_redis_value &&
                          true_redis_value == new_value;
  std::printf("RESULT tier_1_reinsertion_mismatch_reproduced=%s\n",
              reproduced ? "true" : "false");

  wait_for(layer_context);
  CHECK_NVE(nve_context_destroy(redis_context));
  CHECK_NVE(nve_context_destroy(layer_context));
  CHECK_NVE(nve_layer_destroy(layer));
  CHECK_NVE(nve_table_destroy(redis_table));
  CHECK_NVE(nve_table_destroy(host_table));
  CHECK_NVE(nve_table_destroy(gpu_table));
  CHECK_NVE(nve_host_factory_destroy(redis_factory));
  CHECK_NVE(nve_host_factory_destroy(host_factory));
  return reproduced ? 0 : 1;
}
