# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression: flat attention capture and explicit math policy in every mode."""
import json
import copy
import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.convert import CacheSpec, partition_history
from history_cache.precision import configure_math, check_math_policy
from history_cache.single_capture import capture
from history_cache.dataflow_precision import bf16_inputs,audit_dataflow,validate_autocast


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.norm = torch.nn.LayerNorm(64)
        self.register_buffer('query_bias', torch.randn(64) * .01)
        self.force_float = True
        self.key = torch.nn.Linear(8, 256, bias=False)
        self.value = torch.nn.Linear(8, 256, bias=False)

    def forward(self, history, query):
        h = self.embedding(history)
        if self.force_float:
            h, query = h.float(), query.float()
        factory_dtype=torch.float32 if self.force_float else query.dtype
        query=query + torch.zeros_like(query,dtype=factory_dtype) + self.query_bias
        query = self.norm(query)
        k = self.key(h).reshape(h.shape[0], 24, 4, 64).permute(0, 2, 1, 3)
        v = self.value(h).reshape(h.shape[0], 24, 4, 64).permute(0, 2, 1, 3)
        return (query @ k.transpose(-1, -2)).softmax(-1) @ v


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.manual_seed(23)
    model = Attention().cuda().eval()
    inputs = (torch.randint(32, (4, 24), device='cuda'),
              torch.randn(4, 4, 1, 64, device='cuda'))
    results = []
    for precision, tf32, autocast in (('fp32', False, False), ('fp32', True, False),
                                      ('bf16', False, False), ('bf16', False, True)):
        policy = configure_math(precision, tf32)
        check_math_policy(tf32)
        ep, _ = capture(model, inputs, precision,autocast=autocast)
        check_math_policy(tf32)
        assert not any('wrap_with_autocast' in str(n.target) for n in ep.graph.nodes)
        part = partition_history(ep, CacheSpec((0,), 24, 4, 64))
        assert [b['prefix_length'] for b in part.report['banks']] == [24]
        nodes = {n.name: n for n in ep.graph.nodes}
        expected = torch.float32 if precision == 'fp32' else torch.bfloat16
        for bank in part.report['banks']:
            for key in ('key_node', 'value_node'):
                assert nodes[bank[key]].meta['val'].dtype == expected
        pure_bf16=precision=='bf16' and not autocast
        reference=copy.deepcopy(model)
        replay_inputs=inputs
        if pure_bf16:
            reference.to(torch.bfloat16)
            reference.force_float=False
            replay_inputs=bf16_inputs(inputs)
        assert next(model.parameters()).dtype==torch.float32, 'Capture mutated caller weights'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,enabled=autocast):
            expected_output = reference(*replay_inputs)
        with torch.no_grad():
            actual = ep.module()(*replay_inputs)
        torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
        audit=audit_dataflow(ep,require_bf16=pure_bf16)
        if pure_bf16:
            assert not audit['floating_dtype_transitions']
            assert audit['floating_state_dtypes']=={'torch.bfloat16':6}
            assert ep.example_inputs[0][0].dtype==torch.int64
            assert ep.example_inputs[0][1].dtype==torch.bfloat16
        else:
            try:audit_dataflow(ep,require_bf16=True)
            except ValueError:pass
            else:raise AssertionError('FP32/mixed graph incorrectly passed strict BF16 audit')
        results.append({'precision': precision,'autocast':autocast, **policy, 'flat_capture': True,
                        'dtype_audit':audit,
                        'bank_lengths': [24], 'replay_bit_exact': True})
    configure_math('fp32', False)
    try:
        check_math_policy(True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed runtime policy was not rejected')
    for precision, env in (('bf16', {}), ('fp32', {'NVIDIA_TF32_OVERRIDE': '0'})):
        with patch.dict(os.environ, env):
            try:
                configure_math(precision, True)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid TF32 configuration was accepted')
    try:validate_autocast('fp32',True)
    except ValueError:pass
    else:raise AssertionError('FP32 autocast policy accepted')
    out = args.output or Path(__file__).resolve().parents[1] / 'artifacts/single_program/precision_regression.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status': 'passed', 'cases': results,
                              'invalid_policy_rejected': True}, indent=2) + '\n')
    print('PRECISION_REGRESSION_PASS', [{k:v for k,v in row.items() if k!='dtype_audit'}
                                      for row in results], flush=True)


if __name__ == '__main__':
    main()
