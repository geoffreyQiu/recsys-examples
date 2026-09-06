// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace packed_jagged {

using AdapterOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {

constexpr int kThreads = 256;
constexpr int64_t kValuesPerTile = 2048;
constexpr int64_t kMaxGridStrideBlocks = 65535;

template <bool Filtered>
__global__ void prepare_lengths_fm_kernel(
    const int64_t* __restrict__ lengths_rm,
    const int64_t* __restrict__ drop_prefix,
    int64_t* __restrict__ lengths_fm,
    int64_t* __restrict__ src_offsets,
    int64_t* __restrict__ dst_offsets,
    int64_t batch_size,
    int64_t num_features,
    int64_t segment_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    src_offsets[0] = 0;
    dst_offsets[0] = 0;
  }
  for (int64_t output_segment =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output_segment < segment_count;
       output_segment += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t feature = output_segment / batch_size;
    const int64_t request = output_segment - feature * batch_size;
    const int64_t input_segment = request * num_features + feature;
    const int64_t prefix = Filtered ? drop_prefix[input_segment] : 0;
    lengths_fm[output_segment] = lengths_rm[input_segment] - prefix;
  }
}

__device__ __forceinline__ int64_t find_output_segment(
    const int64_t* __restrict__ offsets,
    int64_t segment_count,
    int64_t output_index) {
  int64_t low = 0;
  int64_t high = segment_count + 1;
  while (low < high) {
    const int64_t mid = low + (high - low) / 2;
    if (offsets[mid] <= output_index) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low - 1;
}

template <bool Filtered>
__global__ void copy_output_tiles_kernel(
    const int64_t* __restrict__ values_rm,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ drop_prefix,
    int64_t* __restrict__ values_fm,
    int64_t batch_size,
    int64_t num_features,
    int64_t segment_count,
    int64_t output_numel,
    int64_t num_tiles) {
  __shared__ int64_t first_segment;

  for (int64_t tile_id = blockIdx.x; tile_id < num_tiles;
       tile_id += gridDim.x) {
    const int64_t tile_begin = tile_id * kValuesPerTile;
    const int64_t tile_limit = tile_begin + kValuesPerTile;
    const int64_t tile_end =
        tile_limit < output_numel ? tile_limit : output_numel;

    if (threadIdx.x == 0) {
      first_segment =
          find_output_segment(dst_offsets, segment_count, tile_begin);
    }
    __syncthreads();

    int64_t segment = first_segment;
    int64_t cursor = tile_begin;
    while (cursor < tile_end && segment < segment_count) {
      const int64_t segment_begin = dst_offsets[segment];
      const int64_t segment_end = dst_offsets[segment + 1];
      if (segment_end <= cursor) {
        ++segment;
        continue;
      }

      const int64_t chunk_begin =
          cursor > segment_begin ? cursor : segment_begin;
      const int64_t chunk_end =
          tile_end < segment_end ? tile_end : segment_end;
      const int64_t feature = segment / batch_size;
      const int64_t request = segment - feature * batch_size;
      const int64_t input_segment = request * num_features + feature;
      const int64_t prefix = Filtered ? drop_prefix[input_segment] : 0;
      const int64_t source_begin = src_offsets[input_segment] + prefix;

      for (int64_t output_index = chunk_begin + threadIdx.x;
           output_index < chunk_end;
           output_index += blockDim.x) {
        values_fm[output_index] =
            values_rm[source_begin + output_index - segment_begin];
      }

      cursor = chunk_end;
      ++segment;
    }
    __syncthreads();
  }
}

struct Metadata {
  at::Tensor lengths_fm;
  at::Tensor src_offsets;
  at::Tensor dst_offsets;
};

template <bool Filtered>
Metadata prepare_metadata(
    const at::Tensor& lengths_rm,
    const int64_t* drop_prefix,
    cudaStream_t stream) {
  const int64_t batch_size = lengths_rm.size(0);
  const int64_t num_features = lengths_rm.size(1);
  const int64_t segment_count = batch_size * num_features;
  TORCH_CHECK(
      segment_count <= std::numeric_limits<int>::max(),
      "B * M exceeds the CUB scan item limit");

  auto lengths_fm =
      at::empty({num_features, batch_size}, lengths_rm.options());
  auto src_offsets = at::empty({segment_count + 1}, lengths_rm.options());
  auto dst_offsets = at::empty({segment_count + 1}, lengths_rm.options());

  const int64_t blocks64 =
      (segment_count + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(
      std::min<int64_t>(blocks64, kMaxGridStrideBlocks));
  prepare_lengths_fm_kernel<Filtered><<<blocks, kThreads, 0, stream>>>(
      lengths_rm.data_ptr<int64_t>(),
      drop_prefix,
      lengths_fm.data_ptr<int64_t>(),
      src_offsets.data_ptr<int64_t>(),
      dst_offsets.data_ptr<int64_t>(),
      batch_size,
      num_features,
      segment_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t src_temp_bytes = 0;
  size_t dst_temp_bytes = 0;
  const int scan_items = static_cast<int>(segment_count);
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      nullptr,
      src_temp_bytes,
      lengths_rm.data_ptr<int64_t>(),
      src_offsets.data_ptr<int64_t>() + 1,
      scan_items,
      stream));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      nullptr,
      dst_temp_bytes,
      lengths_fm.data_ptr<int64_t>(),
      dst_offsets.data_ptr<int64_t>() + 1,
      scan_items,
      stream));

  const size_t temp_bytes = std::max(src_temp_bytes, dst_temp_bytes);
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_bytes)},
      lengths_rm.options().dtype(at::kByte));
  void* temp_ptr = temp_bytes == 0 ? nullptr : temp_storage.data_ptr();

  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      temp_ptr,
      src_temp_bytes,
      lengths_rm.data_ptr<int64_t>(),
      src_offsets.data_ptr<int64_t>() + 1,
      scan_items,
      stream));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      temp_ptr,
      dst_temp_bytes,
      lengths_fm.data_ptr<int64_t>(),
      dst_offsets.data_ptr<int64_t>() + 1,
      scan_items,
      stream));

  return {lengths_fm, src_offsets, dst_offsets};
}

template <bool Filtered>
void launch_tiled_copy(
    const at::Tensor& values_rm,
    const at::Tensor& src_offsets,
    const at::Tensor& dst_offsets,
    const int64_t* drop_prefix,
    at::Tensor& values_fm,
    int64_t batch_size,
    int64_t num_features,
    cudaStream_t stream) {
  const int64_t output_numel = values_fm.numel();
  if (output_numel == 0) {
    return;
  }

  const int64_t segment_count = batch_size * num_features;
  const int64_t num_tiles =
      1 + (output_numel - 1) / kValuesPerTile;
  const int blocks = static_cast<int>(
      std::min<int64_t>(num_tiles, kMaxGridStrideBlocks));

  copy_output_tiles_kernel<Filtered><<<blocks, kThreads, 0, stream>>>(
      values_rm.data_ptr<int64_t>(),
      src_offsets.data_ptr<int64_t>(),
      dst_offsets.data_ptr<int64_t>(),
      drop_prefix,
      values_fm.data_ptr<int64_t>(),
      batch_size,
      num_features,
      segment_count,
      output_numel,
      num_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool Filtered>
AdapterOutput run_adapter(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm,
    const int64_t* drop_prefix) {
  c10::cuda::CUDAGuard device_guard(values_rm.device());
  const auto current_stream =
      at::cuda::getCurrentCUDAStream(values_rm.get_device());
  const cudaStream_t stream = current_stream.stream();

  Metadata metadata =
      prepare_metadata<Filtered>(lengths_rm, drop_prefix, stream);
  at::Tensor values_fm;
  if constexpr (Filtered) {
    int64_t output_numel = 0;
    const int64_t segment_count = lengths_rm.numel();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &output_numel,
        metadata.dst_offsets.data_ptr<int64_t>() + segment_count,
        sizeof(output_numel),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(
        output_numel >= 0,
        "filtered output size must be non-negative");
    TORCH_CHECK(
        output_numel <= values_rm.numel(),
        "filtered output cannot exceed the input value count");
    values_fm = at::empty({output_numel}, values_rm.options());
  } else {
    values_fm = at::empty_like(values_rm);
  }

  launch_tiled_copy<Filtered>(
      values_rm,
      metadata.src_offsets,
      metadata.dst_offsets,
      drop_prefix,
      values_fm,
      lengths_rm.size(0),
      lengths_rm.size(1),
      stream);
  return {
      std::move(values_fm),
      std::move(metadata.lengths_fm),
      std::move(metadata.dst_offsets)};
}

} // namespace

AdapterOutput reorder_cuda_impl(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm) {
  return run_adapter<false>(values_rm, lengths_rm, nullptr);
}

AdapterOutput reorder_and_filter_cuda_impl(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm,
    const at::Tensor& drop_prefix) {
  return run_adapter<true>(
      values_rm,
      lengths_rm,
      drop_prefix.data_ptr<int64_t>());
}

} // namespace packed_jagged
