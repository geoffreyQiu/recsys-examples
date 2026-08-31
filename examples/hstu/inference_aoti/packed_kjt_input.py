# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from commons.datasets.hstu_batch import HSTUBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Load native schemas before registering their FakeTensor implementations.
# isort: off
import hstu_cuda_ops  # noqa: F401
import commons.ops.cuda_ops.fake_packed_jagged_ops  # noqa: F401
# isort: on


def kjt_to_request_major(
    features: KeyedJaggedTensor,
    expected_keys: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one concrete feature-major KJT as request-major tensors."""

    keys = tuple(features.keys())
    if keys != tuple(expected_keys):
        raise ValueError("KJT key order does not match the exported model")

    num_features = len(keys)
    lengths = features.lengths()
    if num_features == 0 or lengths.numel() % num_features != 0:
        raise ValueError("KJT lengths must contain one entry per request and key")

    batch_size = lengths.numel() // num_features
    if batch_size == 0:
        raise ValueError("KJT batch size must be positive")

    lengths_fm = lengths.reshape(num_features, batch_size)
    offsets_fm = features.offsets().detach().cpu().tolist()
    values_fm = features.values()
    segments = []
    for request in range(batch_size):
        for feature in range(num_features):
            segment = feature * batch_size + request
            segments.append(values_fm[offsets_fm[segment] : offsets_fm[segment + 1]])
    values_rm = torch.cat(segments)
    return values_rm.contiguous(), lengths_fm.transpose(0, 1).contiguous()


class HSTUPackedInputWrapper(torch.nn.Module):
    """Expose the non-KV HSTU model through plain jagged tensors."""

    def __init__(
        self,
        inner: torch.nn.Module,
        example_batch: HSTUBatch,
        *,
        request_major: bool = True,
    ) -> None:
        super().__init__()
        self.inner = inner
        self._request_major = request_major
        self._keys = tuple(example_batch.features.keys())
        self._contextual_feature_names = list(
            example_batch.contextual_feature_names
        )
        self._item_feature_name = example_batch.item_feature_name
        self._action_feature_name = example_batch.action_feature_name
        self._feature_to_max_seqlen = dict(example_batch.feature_to_max_seqlen)
        self._max_num_candidates = int(example_batch.max_num_candidates)

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if self._request_major:
            values_fm, lengths_fm, offsets_fm = torch.ops.packed_jagged.reorder(
                values,
                lengths,
            )
        else:
            values_fm = values
            lengths_fm = lengths.long()
            offsets_fm = torch.ops.fbgemm.asynchronous_complete_cumsum(
                lengths_fm
            )
        features = KeyedJaggedTensor(
            keys=list(self._keys),
            values=values_fm,
            lengths=lengths_fm.reshape(-1),
            offsets=offsets_fm,
        )
        batch = HSTUBatch(
            features=features,
            # Required positive metadata; runtime B comes from tensor shapes.
            batch_size=1,
            feature_to_max_seqlen=self._feature_to_max_seqlen,
            contextual_feature_names=self._contextual_feature_names,
            actual_batch_size=None,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            max_num_candidates=self._max_num_candidates,
            num_candidates=num_candidates,
        )
        return self.inner(batch).float().cpu()
