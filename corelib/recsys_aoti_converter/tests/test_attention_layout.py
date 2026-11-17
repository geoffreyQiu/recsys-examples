# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU structural/numerical checks for grouped independent attention queries."""
import copy
import json
from pathlib import Path
import sys

import torch
from torch.nn.functional import scaled_dot_product_attention
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.attention_layout import fold_broadcast_attention


class Repeated(torch.nn.Module):
    def __init__(self, groups=5, queries=1, dropout=0., variable_bias=False, causal=False):
        super().__init__()
        self.groups, self.queries = groups, queries
        self.dropout, self.variable_bias, self.causal = dropout, variable_bias, causal

    def forward(self, q, k, v, mask):
        batch, heads, length, width = k.shape
        query = q.reshape(batch * self.groups, heads, self.queries, width)
        key = k.unsqueeze(1).expand(batch, self.groups, heads, length, width).clone().reshape(batch * self.groups, heads, length, width)
        value = v.unsqueeze(1).expand(batch, self.groups, heads, length, width).clone().reshape(batch * self.groups, heads, length, width)
        if self.variable_bias:
            mask = mask.reshape(batch * self.groups, length)
        else:
            mask = mask.unsqueeze(1).expand(batch, self.groups, length).clone().reshape(batch * self.groups, length)
        bias = (1. - mask.to(q.dtype))[:, None, None, :] * -1e9
        if self.causal:
            bias = None
        a = scaled_dot_product_attention(query, key, value, bias, self.dropout, self.causal)
        b = scaled_dot_product_attention(query, key, value, bias, self.dropout, self.causal)
        return a + b


def run():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    rows = []
    for groups, queries in ((3, 1), (5, 2)):
        model = Repeated(groups, queries)
        def inputs(batch, variable=False):
            q = torch.randn(batch, groups, 2, queries, 8)
            k, v = torch.randn(batch, 2, 7, 8), torch.randn(batch, 2, 7, 8)
            mask = torch.randint(0, 2, (batch, groups, 7) if variable else (batch, 7)).bool()
            return q, k, v, mask
        sample = inputs(2)
        dim = torch.export.Dim('batch', min=1, max=4)
        ep = torch.export.export(model, sample, dynamic_shapes=tuple({0: dim} for _ in sample), strict=False)
        module = ep.module()
        report = fold_broadcast_attention(module)
        assert report['matched'] == 2, report
        assert report['invariant_cse']['attention_calls'] == 1, report
        # The transformed module itself is still a single exportable program.
        again = torch.export.export(module, sample, dynamic_shapes=tuple({0: dim} for _ in sample), strict=False)
        for batch in (1, 2, 4):
            for all_masked in (False, True):
                args = inputs(batch)
                if all_masked:
                    args[-1].zero_()
                with torch.no_grad():
                    expected = model(*args)
                    actual = module(*args)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                rows.append({'groups': groups, 'queries': queries, 'batch': batch,
                             'all_masked': all_masked, 'max_abs': (actual - expected).abs().max().item()})
        for opts in ({'variable_bias': True}, {'dropout': .2}, {'causal': True}):
            model = Repeated(groups, queries, **opts)
            args = inputs(2, opts.get('variable_bias', False))
            negative = torch.export.export(model, args, strict=False).module()
            result = fold_broadcast_attention(negative)
            assert result['matched'] == 0, (opts, result)
        mutable = torch.export.export(Repeated(groups, queries), sample, strict=False).module()
        first_attention = next(n for n in mutable.graph.nodes if str(n.target) == 'aten.scaled_dot_product_attention.default')
        key_input = next(n for n in mutable.graph.nodes if n.op == 'placeholder' and n.name == 'k')
        with mutable.graph.inserting_before(first_attention):
            mutable.graph.call_function(torch.ops.aten.add_.Tensor, (key_input, 1.))
        result = fold_broadcast_attention(mutable)
        assert result['matched'] == 0 and 'mutable ATen' in result['skipped']
    print(json.dumps({'passed': True, 'numerical_cases': rows, 'negative_cases': 8}, indent=2))


if __name__ == '__main__':
    run()
