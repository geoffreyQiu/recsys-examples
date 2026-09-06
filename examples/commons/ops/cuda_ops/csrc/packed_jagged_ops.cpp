// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <torch/library.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

#ifdef PACKED_JAGGED_STANDALONE
#include <torch/extension.h>
#endif

namespace packed_jagged {

using AdapterOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {

constexpr std::uint64_t kBatchLogLimit = 64;

bool batch_log_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("PACKED_JAGGED_BATCH_LOG");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

void maybe_log_batch(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm) {
  if (!batch_log_enabled()) {
    return;
  }

  static std::atomic<std::uint64_t> call_count{0};
  const std::uint64_t call =
      call_count.fetch_add(1, std::memory_order_relaxed) + 1;
  if (call > kBatchLogLimit) {
    return;
  }

  std::fprintf(
      stderr,
      "[packed-jagged-batch] call=%llu B=%lld M=%lld T=%lld\n",
      static_cast<unsigned long long>(call),
      static_cast<long long>(lengths_rm.size(0)),
      static_cast<long long>(lengths_rm.size(1)),
      static_cast<long long>(values_rm.numel()));
  std::fflush(stderr);
}

} // namespace

AdapterOutput reorder_cuda_impl(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm);

AdapterOutput reorder_and_filter_cuda_impl(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm,
    const at::Tensor& drop_prefix);

void check_common(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm) {
  TORCH_CHECK(values_rm.is_cuda(), "values_rm must be a CUDA tensor");
  TORCH_CHECK(lengths_rm.is_cuda(), "lengths_rm must be a CUDA tensor");
  TORCH_CHECK(
      values_rm.scalar_type() == at::kLong,
      "values_rm must have dtype int64");
  TORCH_CHECK(
      lengths_rm.scalar_type() == at::kLong,
      "lengths_rm must have dtype int64");
  TORCH_CHECK(values_rm.dim() == 1, "values_rm must have shape [T]");
  TORCH_CHECK(lengths_rm.dim() == 2, "lengths_rm must have shape [B, M]");
  TORCH_CHECK(
      lengths_rm.size(0) > 0 && lengths_rm.size(1) > 0,
      "lengths_rm requires B,M > 0");
  TORCH_CHECK(values_rm.is_contiguous(), "values_rm must be contiguous");
  TORCH_CHECK(lengths_rm.is_contiguous(), "lengths_rm must be contiguous");
  TORCH_CHECK(
      values_rm.device() == lengths_rm.device(),
      "values_rm and lengths_rm must be on the same CUDA device");
}

AdapterOutput reorder_cuda(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm) {
  check_common(values_rm, lengths_rm);
  maybe_log_batch(values_rm, lengths_rm);
  return reorder_cuda_impl(values_rm, lengths_rm);
}

AdapterOutput reorder_and_filter_cuda(
    const at::Tensor& values_rm,
    const at::Tensor& lengths_rm,
    const at::Tensor& drop_prefix) {
  check_common(values_rm, lengths_rm);
  TORCH_CHECK(drop_prefix.is_cuda(), "drop_prefix must be a CUDA tensor");
  TORCH_CHECK(
      drop_prefix.scalar_type() == at::kLong,
      "drop_prefix must have dtype int64");
  TORCH_CHECK(
      drop_prefix.sizes() == lengths_rm.sizes(),
      "drop_prefix must have the same [B, M] shape as lengths_rm");
  TORCH_CHECK(drop_prefix.is_contiguous(), "drop_prefix must be contiguous");
  TORCH_CHECK(
      drop_prefix.device() == values_rm.device(),
      "drop_prefix must be on the same CUDA device as values_rm");
  return reorder_and_filter_cuda_impl(values_rm, lengths_rm, drop_prefix);
}

} // namespace packed_jagged

TORCH_LIBRARY_FRAGMENT(packed_jagged, m) {
  m.def(
      "reorder(Tensor values_rm, Tensor lengths_rm) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "reorder_and_filter(Tensor values_rm, Tensor lengths_rm, "
      "Tensor drop_prefix) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(packed_jagged, CUDA, m) {
  m.impl("reorder", &packed_jagged::reorder_cuda);
  m.impl(
      "reorder_and_filter",
      &packed_jagged::reorder_and_filter_cuda);
}

#ifdef PACKED_JAGGED_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
#endif
