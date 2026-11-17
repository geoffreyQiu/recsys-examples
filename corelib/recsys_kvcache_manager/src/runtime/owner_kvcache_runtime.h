// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <ATen/ATen.h>

#include <memory>

#include "interface_kvcache_runtime.h"

namespace kvcache_manager {
// Standard paged-operator facade. The fixed-slot converter uses CacheOwner
// directly, preserving its single metadata handoff instead of nesting facades.
std::shared_ptr<IKVCacheRuntime> make_owner_kvcache_runtime();
}  // namespace kvcache_manager
