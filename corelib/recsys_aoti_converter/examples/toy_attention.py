# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Small synthetic cross-attention model for the conversion example."""
import math

import torch

INPUT_NAMES = ("history_ids", "queries")


class ToyAttention(torch.nn.Module):
    def __init__(self, vocabulary=37, embedding_dim=24, query_dim=12,
                 heads=2, head_dim=64):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        width = heads * head_dim
        self.embedding = torch.nn.Embedding(vocabulary, embedding_dim)
        self.key = torch.nn.Linear(embedding_dim, width, bias=False)
        self.value = torch.nn.Linear(embedding_dim, width, bias=False)
        self.query = torch.nn.Linear(query_dim, width, bias=False)

    def _heads(self, value):
        return value.reshape(value.shape[0], value.shape[1],
                             self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, inputs):
        history = self.embedding(inputs["history_ids"])
        key, value = self._heads(self.key(history)), self._heads(self.value(history))
        query = self._heads(self.query(inputs["queries"]))
        probabilities = (query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        attended = probabilities @ value
        return attended.transpose(1, 2).reshape(query.shape[0], query.shape[2],
                                               self.heads * self.head_dim)
