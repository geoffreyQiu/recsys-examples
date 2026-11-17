# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Check mutation-aware append values and slot isolation in both storage dtypes."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from history_cache.modules import FixedHistoryFill


@torch.inference_mode()
def main():
    torch.ops.load_library(str(ROOT / "build/append/recsys_paged_kvcache_ops.so"))
    for dtype in (torch.float32, torch.bfloat16):
        table = torch.full((3, 2, 32, 2, 64), 7, dtype=dtype, device="cuda")
        key = torch.randn(2, 2, 32, 64, dtype=dtype, device="cuda")
        value = torch.randn_like(key)
        slots = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
        fill = FixedHistoryFill(length=32, heads=2, head_dim=64, max_batch=2)
        fill(key, value, slots, table)
        actual = table.index_select(0, slots.long())
        torch.testing.assert_close(actual[:, 0].permute(0, 2, 1, 3), key, rtol=0, atol=0)
        torch.testing.assert_close(actual[:, 1].permute(0, 2, 1, 3), value, rtol=0, atol=0)
        assert bool((table[1] == 7).all()), "append modified an unrelated history slot"
        print("APPEND_FILL_PASS", dtype)


if __name__ == "__main__":
    main()
