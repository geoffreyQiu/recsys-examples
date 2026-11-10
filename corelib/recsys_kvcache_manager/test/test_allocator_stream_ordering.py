# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Target-only regression: native metadata writes must follow caller-stream writes.

No model, FlexKV process, or large KV pool is required. This deliberately queues
an earlier writer to the supplied buffers, as can occur with allocator reuse.
"""
import pytest
import torch

kvcache_cpp = pytest.importorskip("kvcache_cpp")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("side_stream", [False, True])
def test_allocate_orders_after_prior_buffer_writer(side_stream):
    device = torch.cuda.current_device()
    stream = torch.cuda.Stream(device=device) if side_stream else torch.cuda.default_stream(device)
    with torch.cuda.stream(stream):
        manager = kvcache_cpp.GPUKVCacheManagerImpl(1, 1, 8, 32, 32, 64, 0, 1, 64, device)
        uids = torch.tensor([10], dtype=torch.int64)
        lengths = torch.tensor([64], dtype=torch.int64)
        host_lengths = torch.zeros(1, dtype=torch.int32)
        pages = torch.empty(2, dtype=torch.int32, device=device)
        metadata = torch.empty(9 + 2 * 64, dtype=torch.int32, device=device)
        stream.synchronize()

        # Leave a previous write pending on the caller stream. Without the
        # incoming dependency, alloc_stream can write first and be clobbered.
        torch.cuda._sleep(50_000_000)
        pages.fill_(-9)
        metadata.fill_(-9)
        manager.allocate(uids, lengths, host_lengths, pages, metadata)
        stream.synchronize()

        assert pages.cpu().tolist() == [0, 1]
        assert metadata[:9].cpu().tolist() == [0, 2, 32, 64, 0, 64, 64, 0, 64]
        assert metadata[9:73].cpu().tolist() == [0] * 64
        assert metadata[73:].cpu().tolist() == list(range(64))
