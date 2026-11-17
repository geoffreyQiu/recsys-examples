# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent modules inserted by the conversion API."""
import torch


class FixedHistoryFill(torch.nn.Module):
    def __init__(self, length, heads, head_dim, max_batch=8, banks=1):
        super().__init__()
        self.length, self.heads, self.head_dim = length, heads, head_dim
        self.banks = banks
        self.max_nnz = max_batch * length
        self.register_buffer('rows', torch.arange(max_batch, device='cuda', dtype=torch.int32).repeat_interleave(length))
        self.register_buffer('positions', torch.arange(length, device='cuda', dtype=torch.int32).repeat(max_batch))
        self.register_buffer('indptr', torch.arange(max_batch+1, device='cuda', dtype=torch.int32))
        self.register_buffer('lengths', torch.full((max_batch,), length, device='cuda', dtype=torch.int32))
        self.register_buffer('offsets', torch.zeros(max_batch+1, device='cuda', dtype=torch.int32))

    def forward(self, key, value, slots, *tables):
        n = key.shape[0]
        nnz = torch.full((1,), n*self.length, dtype=torch.int32, device=key.device)
        for bank in range(self.banks):
            begin, end = bank*self.length, (bank+1)*self.length
            k = key[:, :, begin:end].permute(0, 2, 1, 3).reshape(-1, self.heads, self.head_dim).contiguous()
            v = value[:, :, begin:end].permute(0, 2, 1, 3).reshape(-1, self.heads, self.head_dim).contiguous()
            torch.ops.paged_kvcache_ops.append_kvcache.mutate(
                k, v, self.rows[:n*self.length], self.positions[:n*self.length],
                self.offsets[:n+1], nnz, self.max_nnz, tables[bank], slots,
                self.indptr[:n+1], self.lengths[:n], 0)
        return tables


class CachedConsumer(torch.nn.Module):
    def __init__(self, consumer, banks=1):
        super().__init__()
        self.consumer = consumer
        self.banks = banks

    def forward(self, tasks, slots, *inputs):
        # Readiness is a data dependency of every cache read in the graph.
        tables = torch.ops.kvcache_manager_ops.onboard_wait(tasks, slots)
        keys, values = [], []
        for bank in range(self.banks):
            rows = tables[bank].index_select(0, slots.to(torch.int64))
            keys.append(rows[:, 0].permute(0, 2, 1, 3))
            values.append(rows[:, 1].permute(0, 2, 1, 3))
        key = torch.cat(keys, dim=2).contiguous()
        value = torch.cat(values, dim=2).contiguous()
        return self.consumer(*inputs, key, value)
