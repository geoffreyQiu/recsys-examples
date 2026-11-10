# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Target-only regression using Python's standard-library test runner.

Run this file directly. No pytest, model, FlexKV process, or large KV pool is
required. The test deliberately queues an earlier writer to supplied buffers,
as can occur when PyTorch reuses storage on the caller's current stream.
"""
import unittest

import torch
import kvcache_cpp


class TestAllocatorStreamOrdering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; this validation must not pass by skipping tests")
        marker = getattr(kvcache_cpp, "cache_allocator_stream_ordering", "unmarked")
        if marker != "events_v1":
            raise RuntimeError(f"Expected allocator_impl=events_v1, got {marker}")
        print(f"[AllocatorRegression] allocator_impl={marker}", flush=True)

    def _check_prior_buffer_writer(self, side_stream):
        device = torch.cuda.current_device()
        stream = (
            torch.cuda.Stream(device=device)
            if side_stream else torch.cuda.default_stream(device)
        )
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

            self.assertEqual(pages.cpu().tolist(), [0, 1])
            self.assertEqual(metadata[:9].cpu().tolist(), [0, 2, 32, 64, 0, 64, 64, 0, 64])
            self.assertEqual(metadata[9:73].cpu().tolist(), [0] * 64)
            self.assertEqual(metadata[73:].cpu().tolist(), list(range(64)))

    def test_default_stream(self):
        self._check_prior_buffer_writer(side_stream=False)

    def test_nondefault_current_stream(self):
        self._check_prior_buffer_writer(side_stream=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
