# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Replay a converted package at a chosen batch size using its capture sample."""
import argparse
import json
import os
from pathlib import Path
import sys

# The runner validates the complete ABI, including B=1. Optional generated
# input guards in the tested Torch build can retain the capture's B>=2 guard.
os.environ.setdefault('AOTI_RUNTIME_CHECK_INPUTS', '0')
import torch
from torch.utils import _pytree as pytree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.native import NativeCache
from history_cache.single import SingleRunner


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--snapshot-id-base', type=int, default=1000)
    parser.add_argument('--cpu-roundtrip', action='store_true')
    parser.add_argument('--owner-loop', action=argparse.BooleanOptionalAction, default=True,
                        help='Use the shared corelib cache owner (default)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    max_batch = manifest.get('batch_range', [1, 8])[1]
    if not 1 <= args.batch_size <= max_batch:
        parser.error(f'--batch-size must be within the exported limit: 1..{max_batch}')
    original = torch.export.load(args.bundle / 'original_export.pt2')
    flat = tuple(x.index_select(0, torch.arange(args.batch_size, device=x.device) % x.shape[0])
                 for x in original.example_inputs[0])
    positional, keywords = pytree.tree_unflatten(flat, pytree.treespec_loads(manifest['input_tree']))
    del original
    spec = manifest['spec']
    dtype = torch.bfloat16 if manifest['precision'] == 'bf16' else torch.float32
    with NativeCache(root, args.reference, capacity=manifest['capacity'],
                     banks=manifest['bank_count'], dtype=dtype, owner_loop=args.owner_loop,
                     max_batch=manifest.get('batch_range', [1, 8])[1],
                     **{k: spec[k] for k in ('length', 'heads', 'head_dim')}) as cache:
        runner = SingleRunner(args.bundle, cache)
        ids = list(range(args.snapshot_id_base, args.snapshot_id_base + args.batch_size))
        for phase in ('cold', 'gpu_hit'):
            output = runner(*positional, **keywords, snapshot_ids=ids)
            print(json.dumps({'phase': phase, 'stats': runner.last_stats,
                              'output_shapes': [list(x.shape) for x in pytree.tree_leaves(output)]}))
        if args.cpu_roundtrip:
            saved = pytree.tree_map(lambda x: x.clone(), output)
            runner(*positional, **keywords, snapshot_ids=ids, offload=True)
            cache.flush()
            cache.evict(ids)
            output = runner(*positional, **keywords, snapshot_ids=ids)
            assert runner.last_stats['cpu_hits'] == args.batch_size
            for a, b in zip(pytree.tree_leaves(output), pytree.tree_leaves(saved)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            print(json.dumps({'phase': 'cpu_hit', 'stats': runner.last_stats, 'output_bit_exact': True}))


if __name__ == '__main__':
    main()
