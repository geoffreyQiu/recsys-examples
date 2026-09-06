# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FakeTensor registrations for the native packed-jagged operators."""

import torch
from torch import Tensor


def _check_inputs(values_rm: Tensor, lengths_rm: Tensor) -> tuple[int, int]:
    torch._check(values_rm.dim() == 1)
    torch._check(lengths_rm.dim() == 2)
    torch._check(values_rm.dtype == torch.int64)
    torch._check(lengths_rm.dtype == torch.int64)
    torch._check(values_rm.device == lengths_rm.device)
    batch_size, num_features = lengths_rm.shape
    return batch_size, num_features


@torch.library.register_fake("packed_jagged::reorder")
def _reorder_fake(
    values_rm: Tensor,
    lengths_rm: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, num_features = _check_inputs(values_rm, lengths_rm)
    return (
        values_rm.new_empty(values_rm.shape),
        lengths_rm.new_empty((num_features, batch_size)),
        lengths_rm.new_empty((num_features * batch_size + 1,)),
    )


@torch.library.register_fake("packed_jagged::reorder_and_filter")
def _reorder_and_filter_fake(
    values_rm: Tensor,
    lengths_rm: Tensor,
    drop_prefix: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, num_features = _check_inputs(values_rm, lengths_rm)
    torch._check(drop_prefix.dim() == 2)
    torch._check(drop_prefix.dtype == torch.int64)
    torch._check(drop_prefix.device == values_rm.device)
    torch._check(drop_prefix.shape == lengths_rm.shape)
    output_numel = torch.library.get_ctx().new_dynamic_size()
    return (
        values_rm.new_empty((output_numel,)),
        lengths_rm.new_empty((num_features, batch_size)),
        lengths_rm.new_empty((num_features * batch_size + 1,)),
    )
