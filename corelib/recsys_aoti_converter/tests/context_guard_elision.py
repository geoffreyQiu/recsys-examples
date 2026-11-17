# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only structural tests for immutable-history context-guard elision.

No CUDA execution, native cache library, external weights or input data are
required. The fixtures export real repeated HSTU stacks, so the test covers the
analysis/report/rewrite integration rather than just a Boolean helper.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.convert import CacheSpec
from history_cache.layered import analyze_layered, rewrite_layered
from history_cache.context_guard import UniformSlice, UNKNOWN


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(128)
        self.uvqk = torch.nn.Linear(128, 512)
        self.norm2 = torch.nn.LayerNorm(64)
        self.linear = torch.nn.Linear(128, 128)

    def forward(self, x, valid, mask):
        u, v, q, k = self.uvqk(self.norm1(x)).split([128] * 4, -1)
        u = F.silu(u.reshape(-1, 21, 2, 64).contiguous())
        v = v.reshape(-1, 21, 2, 64).contiguous()
        q = q.reshape(-1, 21, 2, 64).contiguous()
        k = k.reshape(-1, 21, 2, 64).contiguous()
        weights = F.silu(torch.einsum('bqhd,bkhd->bhqk', q, k) * .125)
        weights = weights / valid.sum(1).clamp(min=1).reshape(-1, 1, 1, 1)
        weights = weights * mask.unsqueeze(1)
        attended = torch.einsum('bhqk,bkhv->bhqv', weights, v)
        gated = u.transpose(1, 2) * self.norm2(attended)
        return self.linear(gated.transpose(1, 2).reshape(-1, 21, 128)) + x


class ContextStack(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.embed_history = torch.nn.Linear(7, 128)
        self.embed_candidate = torch.nn.Linear(9, 128)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(2)])
        mask = torch.zeros(21, 21)
        mask[:5, :5] = torch.eye(5)
        mask[:5, 5:] = 1
        mask[5:, 5:] = torch.ones(16, 16).triu()
        self.register_buffer('structural_mask', mask)

    def validity(self, history, candidates, history_marker, candidate_marker):
        # new_full's template values do not determine its output. This mirrors
        # a shape-only template dependency before a validity mask.
        template = candidate_marker.unsqueeze(0)
        fill = -1.5 if self.mode == 'narrowing_cast' else 1.0
        if self.mode == 'varying_candidates':
            fill = -1.0
        marker = template.new_full((1, history.shape[0], 5), fill).unbind(0)[0]
        if self.mode == 'narrowing_cast':
            # -1.5 -> int64 -> float32 is -1. Dropping casts would falsely
            # prove every candidate valid and incorrectly eliminate the guard.
            marker = marker.to(torch.int64).to(torch.float32)
        if self.mode == 'history_from_candidates':
            history_marker = torch.cat([candidate_marker[:, :1], history_marker[:, 1:]], 1)
        left = torch.cat([marker, history_marker], 1)[:, :21]
        right = torch.cat([candidate_marker, history_marker], 1)[:, :21]
        # Exercise a constructed sentinel rather than a literal comparison operand.
        sentinel = torch.zeros_like(left) + (-1)
        return ~((left == sentinel) & (right == sentinel))

    def forward(self, history, candidates, history_marker, candidate_marker):
        valid = self.validity(history, candidates, history_marker, candidate_marker).to(history.dtype)
        mask = self.structural_mask.unsqueeze(0).repeat(history.shape[0], 1, 1)
        mask = mask * (valid.unsqueeze(-1) @ valid.unsqueeze(1))
        result = []
        for step in range(2):
            combined = torch.cat([candidates[:, 0, :, :7], history], 1)
            h = self.embed_history(combined[:, 5:])
            c = self.embed_candidate(candidates[:, step])
            x = torch.cat([c, h], 1)
            for layer in self.layers:
                x = layer(x, valid, mask)
            result.append(x[:, :5])
        return torch.stack(result, 1)


def define_rewrite_schemas():
    """Register schemas only; rewritten graphs are inspected, never executed."""
    library = torch.library.Library('history_single', 'FRAGMENT')
    schemas = {
        'prepare': 'prepare(Tensor ids, Tensor anchor, int length, int banks, Tensor? context=None) -> (Tensor, Tensor, Tensor)',
        'pack': 'pack(Tensor[] inputs, Tensor rows, Tensor slots, Tensor ticket) -> Tensor[]',
        'fill': 'fill(Tensor pool, Tensor slots, Tensor key, Tensor value, Tensor ticket, int bank) -> Tensor',
        'onboard_wait': 'onboard_wait(Tensor ticket, Tensor filled, Tensor ordering) -> Tensor',
        'gather': 'gather(Tensor pool, Tensor slots, Tensor ready, int bank) -> (Tensor, Tensor)',
        'offload_launch': 'offload_launch(Tensor ticket, Tensor ordering, Tensor enabled) -> Tensor',
    }
    for name, schema in schemas.items():
        if not hasattr(torch.ops.history_single, name):
            library.define(schema)
    return library


def broadcast_uniformity_regression():
    class BroadcastMask(torch.nn.Module):
        def forward(self, values):
            # The first channel is always false, but the second remains data
            # dependent. Proving only channel0 must not imply uniformity over
            # the whole broadcast operand or eliminate a guard for channel1.
            mask = torch.cat([torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)])
            return (values & mask).unbind(-1)[1]
    model = BroadcastMask()
    ep = torch.export.export(model, (torch.ones(4, 5, 2, dtype=torch.bool),))
    graph = ep.module().graph
    output = next(n for n in graph.nodes if n.op == 'output').args[0][0]
    assert UniformSlice()(output, 1, 0, 5) is UNKNOWN
    zeros = torch.zeros(4, 5, 2, dtype=torch.bool)
    ones = torch.ones_like(zeros)
    assert not torch.equal(model(zeros), model(ones))
    print('PASS broadcast_nonuniform_operand', flush=True)


def constructed_sentinel_regression():
    class SentinelComparison(torch.nn.Module):
        def __init__(self, alpha):
            super().__init__()
            self.alpha = alpha

        def forward(self, values):
            ones = torch.ones_like(values)
            sentinel = torch.add(torch.zeros_like(values), -1, alpha=self.alpha)
            return ones == sentinel

    for alpha, expected in ((1, False), (-1, True)):
        model = SentinelComparison(alpha)
        values = torch.randn(4, 5)
        ep = torch.export.export(model, (values,))
        output = next(n for n in ep.module().graph.nodes if n.op == 'output').args[0][0]
        assert UniformSlice()(output, 1, 0, 5) is expected
        assert bool((model(values) == expected).all())
    print('PASS constructed_sentinel_and_alpha', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(2011)
    torch.set_num_threads(2)
    schemas = define_rewrite_schemas()  # keep registration alive until exit
    broadcast_uniformity_regression()
    constructed_sentinel_regression()
    history = torch.randn(4, 16, 7)
    candidates = torch.randn(4, 2, 5, 9)
    hm = torch.ones(4, 16)
    hm[:, 3] = -1
    cm = torch.ones(4, 5)
    cm[:, 1] = -1
    inputs = (history, candidates, hm, cm)
    spec = CacheSpec((0, 2), 16, 2, 64, history_start=5)
    cases = []
    for mode, expect_elided in (
        ('constant_candidates', True),
        ('varying_candidates', False),
        ('history_from_candidates', False),
        ('narrowing_cast', False),
    ):
        model = ContextStack(mode).eval()
        dim = torch.export.Dim('batch', min=2, max=8)
        with torch.no_grad():
            ep = torch.export.export(model, inputs, dynamic_shapes=tuple({0: dim} for _ in inputs))
        plan = analyze_layered(ep, spec)
        assert plan is not None
        proof = plan.report['context_guard_elision']
        assert proof['elided'] is expect_elided, (mode, proof)
        assert bool(plan.report.get('context_guard')) is not expect_elided, (mode, plan.report)
        converted, report = rewrite_layered(plan, spec)
        prepares = [n for n in converted.graph.nodes if str(n.target) == 'history_single.prepare.default']
        assert len(prepares) == 1
        context = prepares[0].args[4] if len(prepares[0].args) > 4 else None
        assert (context is None) is expect_elided, (mode, context)
        assert report['context_guard_elision'] == proof
        # Verify fixture semantics on different runtime batch/marker values.
        for batch in (1, 4, 8):
            rows = torch.arange(batch) % 4
            h, c, hv, cv = (t.index_select(0, rows) for t in inputs)
            original_valid = model.validity(h, c, hv, cv)
            changed = cv.clone()
            changed[:, 1] = 1
            changed_valid = model.validity(h, c, hv, changed)
            if mode == 'constant_candidates':
                assert bool(original_valid[:, :5].all())
                assert torch.equal(original_valid, changed_valid)
            elif mode in ('varying_candidates', 'narrowing_cast'):
                assert not torch.equal(original_valid[:, :5], changed_valid[:, :5])
            elif mode == 'history_from_candidates':
                changed[:, 0] = -1
                changed_valid = model.validity(h, c, hv, changed)
                assert not torch.equal(original_valid[:, 5:], changed_valid[:, 5:])
        cases.append({'case': mode, 'passed': True, 'proof': proof, 'prepare_context_removed': context is None})
        print('PASS', mode, json.dumps(proof), flush=True)
    result = {'passed': True, 'scope': 'CPU export, static analysis and rewrite; no CUDA/model benchmark',
              'broadcast_nonuniform_operand': 'passed',
              'constructed_sentinel_and_alpha': 'passed', 'cases': cases}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('CONTEXT_GUARD_ELISION_TESTS_PASSED', len(cases), flush=True)


if __name__ == '__main__':
    main()
