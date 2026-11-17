# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Automatic immutable-history graph partitioning and fixed-slot runtime."""
from .convert import CacheSpec, ConversionError, partition_history

__all__ = ["CacheSpec", "ConversionError", "partition_history"]
