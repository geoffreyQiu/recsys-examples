# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Conservative ATen attention-to-SDPA rewrite, independent of model classes.

Run after history matching. Preserve the actual additive-bias expression and
its dtype; never turn finite masking into boolean/-inf masking. SDPA changes
floating-point accumulation/rounding, so mathematical equivalence is not a
promise of bitwise or greedy-decision equivalence.
"""
import math

import torch
from torch.fx import Node


MATMUL = {'aten.matmul.default', 'aten.bmm.default'}
CAST = {'aten._to_copy.default', 'aten.to.dtype'}


def _op(node, names):
    return isinstance(node, Node) and node.op == 'call_function' and str(node.target) in names


def _val(node):
    return node.meta.get('val') if isinstance(node, Node) else None


def _scaled_qk(node):
    scale, chain = 1.0, []
    if _op(node, {'aten.div.Tensor', 'aten.div.Scalar', 'aten.mul.Tensor', 'aten.mul.Scalar'}):
        if len(node.args) != 2 or node.kwargs:
            return None
        base, factor = node.args
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor <= 0:
            return None
        scale = 1.0 / factor if str(node.target).startswith('aten.div.') else float(factor)
        chain.append(node)
        node = base
    if not _op(node, MATMUL) or len(node.args) != 2 or node.kwargs:
        return None
    q, kt = node.args
    if not _op(kt, {'aten.transpose.int'}) or set(kt.args[1:]) != {-1, -2}:
        return None
    return q, kt.args[0], scale, [node, *chain]


def _match(output):
    if not _op(output, MATMUL) or len(output.args) != 2 or output.kwargs:
        return None, None
    weights, value = output.args
    casts = []
    while _op(weights, CAST):
        casts.append(weights)
        weights = weights.args[0]
    if not _op(weights, {'aten.softmax.int'}):
        return None, None
    softmax = weights
    score = softmax.args[0]
    score_meta = _val(score)
    dim = softmax.args[1] if len(softmax.args) > 1 else softmax.kwargs.get('dim')
    if not isinstance(score_meta, torch.Tensor) or dim not in (-1, score_meta.ndim - 1):
        return None, 'softmax must normalize the last dimension'
    bias, add = None, None
    parsed = _scaled_qk(score)
    if parsed is None and _op(score, {'aten.add.Tensor'}):
        if len(score.args) != 2 or score.kwargs.get('alpha', 1) != 1:
            return None, 'unsupported score addition'
        matches = [(i, _scaled_qk(x)) for i, x in enumerate(score.args)]
        matches = [(i, x) for i, x in matches if x is not None]
        if len(matches) == 1:
            i, parsed = matches[0]
            bias, add = score.args[1-i], score
    if parsed is None:
        return None, 'unsupported QK scaling or score transform'
    q, k, scale, score_chain = parsed
    tensors = [_val(x) for x in (q, k, value, output)]
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in tensors):
        return None, 'requires four-dimensional Q/K/V and output metadata'
    qv, kv, vv, ov = tensors
    if qv.dtype not in (torch.float32, torch.bfloat16) or any(x.dtype != qv.dtype for x in tensors):
        return None, 'Q/K/V/output dtypes must agree (FP32 or BF16)'
    if any(x.device != qv.device for x in tensors):
        return None, 'Q/K/V devices must agree'
    if qv.shape[-1] != kv.shape[-1] or kv.shape[-2] != vv.shape[-2]:
        return None, 'incompatible attention dimensions'
    if qv.shape[-3] != kv.shape[-3] or kv.shape[-3] != vv.shape[-3]:
        return None, 'grouped-query attention is not matched'
    for cast in casts:
        cv = _val(cast)
        if not isinstance(cv, torch.Tensor) or cv.dtype != qv.dtype:
            return None, 'unsupported probability cast chain'
        if cast.kwargs.get('device', qv.device) != qv.device:
            return None, 'probability cast moves devices'
    softmax_dtype = softmax.args[2] if len(softmax.args) > 2 else softmax.kwargs.get('dtype')
    if softmax_dtype not in (None, qv.dtype, torch.float32):
        return None, 'unsupported softmax precision'
    if bias is not None:
        bv = _val(bias)
        if not isinstance(bv, torch.Tensor) or bv.dtype != qv.dtype or bv.device != qv.device:
            return None, 'requires an existing floating additive bias matching Q dtype/device'
    chain = [*score_chain, *([add] if add is not None else []), softmax, *casts]
    allowed = set(chain) | {output}
    if any(any(user not in allowed for user in n.users) for n in chain):
        return None, 'scores/probabilities have additional consumers'
    return {'q': q, 'k': k, 'v': value, 'bias': bias, 'scale': scale,
            'chain': chain, 'softmax': softmax}, None


def replace_attention(module):
    """Mutate a GraphModule; return every matched/skipped attention diagnostic."""
    from .review import merge_origins
    graph = module.graph
    replaced, skipped = [], []
    for output in list(graph.nodes):
        match, reason = _match(output)
        if match is None:
            if reason:
                skipped.append({'output': output.name, 'reason': reason})
            continue
        q, k, v, bias = (match[x] for x in ('q', 'k', 'v', 'bias'))
        with graph.inserting_before(output):
            new = graph.call_function(torch.ops.aten.scaled_dot_product_attention.default,
                (q, k, v, bias, 0.0, False), {'scale': match['scale']})
        new.meta = dict(output.meta)
        new.meta['review_id'] = 'sdpa_' + output.meta.get('review_id', output.name)
        new.meta['review_role'] = 'attention_sdpa'
        for source in match['chain']:
            merge_origins(new, source)
        replaced.append({'original_output': output.name, 'replacement': new.name,
                         'original_nodes': [x.name for x in match['chain']] + [output.name],
                         'q': q.name, 'k': k.name, 'v': v.name,
                         'additive_bias': bias.name if bias is not None else None,
                         'bias_dtype': str(_val(bias).dtype) if bias is not None else None,
                         'scale': match['scale'], 'dropout_p': 0.0, 'is_causal': False,
                         'dtype': str(_val(q).dtype)})
        output.replace_all_uses_with(new)
        graph.erase_node(output)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return {'enabled': True, 'matched': len(replaced), 'replacements': replaced,
            'skipped': skipped, 'mask_policy': 'preserve original floating additive bias, including finite masking',
            'numeric_policy': 'SDPA accumulation and removed BF16 intermediate rounding can change scores and decisions'}
