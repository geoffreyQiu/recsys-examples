# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generic sliding-window conversion, dynamic mixed batches, and CPU transfers.

All tensors and weights are generated locally; no external inputs are required.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('AOTI_RUNTIME_CHECK_INPUTS', '0')
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.convert import CacheSpec, ConversionError, partition_history
from history_cache.native import NativeCache
from history_cache.precision import configure_math, TOLERANCES
from history_cache.single import convert, SingleRunner
from history_cache.single_capture import capture
from history_cache.dataflow_precision import bf16_inputs


class WindowAttention(torch.nn.Module):
    def __init__(self, stride=1, unsafe=None):
        super().__init__()
        self.stride, self.unsafe = stride, unsafe
        self.position_scale = torch.nn.Parameter(torch.randn(16, 8) * .1 + 1)
        self.position_bias = torch.nn.Parameter(torch.randn(16, 8) * .1)
        self.key = torch.nn.Linear(8, 128, bias=False)
        self.value = torch.nn.Linear(8, 128, bias=False)

    def forward(self, history, feedback, query):
        outputs = []
        for step in range(3):
            h = history * self.position_scale + self.position_bias
            if self.unsafe == 'token_mix':
                h = h + h.mean(1, keepdim=True)
            if self.unsafe == 'batch_mix':
                h = h + h.mean(0, keepdim=True)
            if self.unsafe == 'request_broadcast':
                h = h + feedback.mean(1, keepdim=True)
            k = self.key(h).reshape(h.shape[0], 16, 2, 64).permute(0, 2, 1, 3)
            v = self.value(h).reshape(h.shape[0], 16, 2, 64).permute(0, 2, 1, 3)
            # Force a layout independent of B; contiguous() specializes this
            # selected view on B!=1 in the pinned Torch export build.
            q = query[:, step].clone(memory_format=torch.contiguous_format)
            outputs.append((q @ k.transpose(-1, -2) / 8).softmax(-1) @ v)
            if step < 2:
                history = torch.cat([history[:, self.stride:],
                                     feedback[:, step*self.stride:(step+1)*self.stride]], dim=1)
        return torch.stack(outputs, dim=1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    p.add_argument('--autocast',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output directory')
    torch.manual_seed(113)
    torch.set_num_threads(4)
    configure_math(a.precision, False)
    root = Path(__file__).resolve().parents[1]
    spec = CacheSpec((0,), 16, 2, 64)
    base = (torch.randn(4, 16, 8, device='cuda') * .1,
            torch.randn(4, 6, 8, device='cuda') * .1,
            torch.randn(4, 3, 2, 4, 64, device='cuda') * .1)
    diagnostics = {}
    for stride in (1, 2, 3):
        model = WindowAttention(stride).eval().cuda()
        ep, _ = capture(model, base, a.precision,autocast=a.autocast)
        part = partition_history(ep, spec)
        assert [b['prefix_length'] for b in part.report['banks']] == [16, 16-stride, 16-2*stride]
        for unsafe in ('token_mix', 'batch_mix', 'request_broadcast'):
            bad, _ = capture(WindowAttention(stride, unsafe).eval().cuda(), base, a.precision,autocast=a.autocast)
            try:
                partition_history(bad, spec)
            except ConversionError as error:
                diagnostics[f'{stride}:{unsafe}'] = str(error)
            else:
                raise AssertionError(f'Unsafe pattern accepted: {stride}/{unsafe}')
    model = WindowAttention().eval().cuda()
    original, tree = capture(model, base, a.precision,autocast=a.autocast)
    dtype = torch.bfloat16 if a.precision == 'bf16' else torch.float32
    if a.precision=='bf16' and not a.autocast:
        model.to(dtype)
        base=bf16_inputs(base)
    cases = []
    with NativeCache(root, a.reference, length=16, heads=2, head_dim=64,
                     banks=3, dtype=dtype) as cache:
        manifest = convert(original, spec, a.output, cache=cache, input_tree=tree,
                           precision=a.precision,autocast=a.autocast,sdpa=True, review_diff=True)
        assert manifest['attention_replacements'] == 3
        runner = SingleRunner(a.output, cache)
        def read_snapshot(identities):
            # Read through the same owner-backed request lifecycle as serving.
            # NativeCache.prepare is the legacy direct-runtime helper.
            op = torch.ops.history_single
            ticket, _, slots = op.prepare(torch.tensor(identities), cache.table,
                                          cache.length, cache.banks)
            assert op.prepared_info(ticket)[1].item() == 0
            empty = cache.table.new_empty((0, cache.heads, cache.length, cache.head_dim))
            values = []
            for bank, pool in enumerate(cache.tables):
                filled = op.fill(pool, slots[:0], empty, empty, ticket, bank)
                ready = op.onboard_wait(ticket, filled, pool)
                values.extend(x.clone() for x in op.gather(pool, slots, ready, bank))
            op.retire(op.offload_launch(ticket, values[-1], torch.tensor([False])))
            return values
        def reference(*inputs):
            with torch.autocast('cuda', dtype=torch.bfloat16,enabled=a.autocast):
                return model(*inputs)
        serial = 10
        for batch in (1, 2, 4, 8):
            inputs = tuple(x.index_select(0, torch.arange(batch, device='cuda') % 4) for x in base)
            ids = list(range(serial, serial+batch)); serial += 10
            def check(label, current, identities, misses, cpu_hits=0, offload=False):
                actual = runner(*current, snapshot_ids=identities, offload=offload)
                torch.testing.assert_close(actual, reference(*current), **TOLERANCES[a.precision])
                assert runner.last_stats['misses'] == misses
                assert runner.last_stats['cpu_hits'] == cpu_hits
                cases.append({'batch': len(identities), 'case': label, 'stats': runner.last_stats.copy()})
                return actual.clone()
            check('cold', inputs, ids, batch)
            check('all_hit', inputs, ids, 0)
            mixed = [identity if i % 2 == 0 else identity+10000 for i, identity in enumerate(ids)]
            check('mixed', inputs, mixed, batch//2)
            shared = (inputs[0][:1].expand_as(inputs[0]).clone(), *inputs[1:])
            dupids = [serial]*batch; serial += 10
            check('shared_history', shared, dupids, 1)
            before = [x.clone() for x in cache.tables]
            changed = (shared[0], shared[1]+.15, shared[2]-.1)
            check('changed_feedback', changed, dupids, 0)
            for x, y in zip(before, cache.tables):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            saved = check('offload', changed, dupids, 0, offload=True)
            cache.flush()
            saved_kv = read_snapshot(dupids[:1])
            cache.evict(dupids[:1])
            restored = check('cpu_hit', changed, dupids, 0, cpu_hits=1)
            torch.testing.assert_close(restored, saved, rtol=0, atol=0)
            restored_kv = read_snapshot(dupids[:1])
            for previous, actual in zip(saved_kv, restored_kv):
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        bad = (shared[0]+1, shared[1], shared[2])
        try:
            runner(*bad, snapshot_ids=dupids)
        except ValueError as error:
            assert 'immutable_history_changed' in str(error)
        else:
            raise AssertionError('Changed history accepted under the same identity')
    (a.output/'test_summary.json').write_text(json.dumps({'status': 'passed','precision':a.precision,
        'autocast':a.autocast,'cases': cases,
        'unsafe_patterns_rejected': diagnostics, 'changed_history_rejected': True}, indent=2)+'\n')
    print('SINGLE_CONVERSION_PASS', a.precision, len(cases), 'runtime cases,', len(diagnostics), 'unsafe graphs')


if __name__ == '__main__':
    main()
