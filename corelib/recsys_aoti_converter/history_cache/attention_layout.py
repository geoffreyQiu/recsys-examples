# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fold candidate-broadcast K/V into the query axis of inference SDPA.

Recognizes structure, not model names: K/V [B,H,L,D] are unsqueezed,
expanded to [B,N,H,L,D], copied and flattened to [B*N,H,L,D]. Independent
queries [B*N,H,Q,D] can instead use [B,H,N*Q,D] and the original K/V.
Only a query-independent bias with the same proven broadcast is accepted.
"""
import operator

import torch
from torch.fx import Node
from .review import merge_origins


def _op(node, names):
    return isinstance(node, Node) and node.op == 'call_function' and str(node.target) in names


def _shape(node):
    value = node.meta.get('val') if isinstance(node, Node) else None
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _same(a, b):
    return str(a) == str(b)


def _writes(node):
    schema = getattr(node.target, '_schema', None)
    if schema is None:
        return False
    return any(arg.alias_info is not None and arg.alias_info.is_write for arg in schema.arguments)


def _repeat(node):
    """Return original tensor and static N for the exact flattened broadcast."""
    if not _op(node, {'aten.view.default', 'aten.reshape.default', 'aten._unsafe_view.default'}):
        return None
    value = node.args[0]
    snapshot = node
    while _op(value, {'aten.clone.default', 'aten.contiguous.default'}):
        snapshot = value
        value = value.args[0]
    if not _op(value, {'aten.expand.default'}):
        return None
    expanded = _shape(value)
    source = value.args[0]
    if not _op(source, {'aten.unsqueeze.default'}) or source.args[1] != 1:
        return None
    base = source.args[0]
    before, after = _shape(base), _shape(node)
    if before is None or after is None or expanded is None:
        return None
    count = value.args[1][1]
    if not isinstance(count, int) or count <= 1 or len(before) != len(after):
        return None
    if (len(expanded) != len(before) + 1 or not _same(after[0], before[0] * count)
            or any(not _same(x, y) for x, y in zip(before[1:], after[1:]))
            or any(not _same(x, y) for x, y in zip(before, (expanded[0], *expanded[2:])))):
        return None
    return base, count, snapshot


_BIAS_POINTWISE = {'aten._to_copy.default', 'aten.to.dtype', 'aten.mul.Tensor',
    'aten.mul.Scalar', 'aten.sub.Tensor', 'aten.add.Tensor', 'aten.div.Tensor',
    'aten.div.Scalar', 'aten.rsub.Scalar', 'aten.neg.default'}


def _bias_plan(node, count, batch):
    """Prove a bias is identical across repeated candidate groups before editing."""
    repeat = _repeat(node)
    if repeat is not None:
        base, n, snapshot = repeat
        return ('base', base, snapshot) if n == count and _same(_shape(base)[0], batch) else None
    if _op(node, {'aten.unsqueeze.default'}):
        if node.args[1] <= 0:
            return None
        parent = _bias_plan(node.args[0], count, batch)
        return ('call', node, [parent]) if parent is not None else None
    if _op(node, _BIAS_POINTWISE):
        plans = []
        for arg in node.args:
            if not isinstance(arg, Node):
                plans.append(('literal', arg))
            else:
                shape = _shape(arg)
                if shape is not None and len(shape) == 0:
                    plans.append(('literal', arg))
                else:
                    plans.append(_bias_plan(arg, count, batch))
        if all(plan is not None for plan in plans):
            return ('call', node, plans)
    return None


def _bias_snapshots(plan):
    if plan is None or plan[0] == 'literal':
        return []
    if plan[0] == 'base':
        return [plan[2]]
    return [node for child in plan[2] for node in _bias_snapshots(child)]


def _deduplicate(module):
    """CSE only immutable inference arithmetic, with barriers at cache effects."""
    pure = {'aten.rsub.Scalar', 'aten.neg.default', 'aten.mul.Tensor', 'aten.mul.Scalar',
        'aten.add.Tensor', 'aten.sub.Tensor', 'aten.div.Tensor', 'aten.div.Scalar',
        'aten._to_copy.default', 'aten.to.dtype', 'aten.unsqueeze.default',
        'aten.reshape.default', 'aten.view.default', 'aten._unsafe_view.default',
        'aten.permute.default', 'aten.transpose.int', 'aten.linear.default', 'aten.sym_size.int'}
    def freeze(value):
        if isinstance(value, (list, tuple)):
            return tuple(freeze(x) for x in value)
        if isinstance(value, dict):
            return tuple((k, freeze(v)) for k, v in sorted(value.items()))
        return value
    seen, removed, attentions = {}, 0, 0
    for node in list(module.graph.nodes):
        name = str(node.target)
        # ATen is functional at this boundary. Opaque/custom ops remain barriers:
        # no read may be hoisted through a fill/wait/offload or unknown effect.
        if node.op in ('call_module', 'call_method') or (node.op == 'call_function'
                and not name.startswith('aten.') and node.target not in (operator.getitem, operator.mul)):
            seen.clear()
        is_attention = name == 'aten.scaled_dot_product_attention.default'
        dropout = node.args[4] if is_attention and len(node.args) > 4 else node.kwargs.get('dropout_p', 0.)
        if node.op != 'call_function' or not (name in pure or node.target is operator.mul or (is_attention and dropout == 0.)):
            continue
        key = (node.target, freeze(node.args), freeze(node.kwargs))
        if key in seen:
            merge_origins(seen[key], node)
            node.replace_all_uses_with(seen[key])
            module.graph.erase_node(node)
            removed += 1
            attentions += int(is_attention)
        else:
            seen[key] = node
    return {'nodes': removed, 'attention_calls': attentions}


def fold_broadcast_attention(module):
    graph = module.graph
    # A clone may snapshot a value before a later in-place write. This initial
    # pass accepts functional exported ATen only, and does not discard such copies.
    if any(n.op == 'call_function' and str(n.target).startswith('aten.') and _writes(n) for n in graph.nodes):
        return {'enabled': True, 'matched': 0, 'skipped': 'graph contains mutable ATen operations'}
    known_cache = {'history_single.prepare.default', 'history_single.pack.default',
        'history_single.fill.default', 'history_single.onboard_wait.default',
        'history_single.gather.default', 'history_single.offload_launch.default'}
    if any(n.op == 'call_function' and getattr(n.target, '_schema', None) is not None
            and not str(n.target).startswith('aten.') and str(n.target) not in known_cache for n in graph.nodes):
        return {'enabled': True, 'matched': 0, 'skipped': 'unrecognized custom operator effect contract'}
    from .slicing import eliminate_common_subexpressions
    normalized = eliminate_common_subexpressions(graph)
    replacements = []
    positions = {n: i for i, n in enumerate(graph.nodes)}
    effects = [positions[n] for n in graph.nodes if n.op in ('call_module', 'call_method') or
        (n.op == 'call_function' and getattr(n.target, '_schema', None) is not None
         and not str(n.target).startswith('aten.'))]
    for node in list(graph.nodes):
        if not _op(node, {'aten.scaled_dot_product_attention.default'}):
            continue
        q, k, v = node.args[:3]
        bias = node.args[3] if len(node.args) > 3 else node.kwargs.get('attn_mask')
        dropout = node.args[4] if len(node.args) > 4 else node.kwargs.get('dropout_p', 0.)
        causal = node.args[5] if len(node.args) > 5 else node.kwargs.get('is_causal', False)
        if dropout != 0. or causal or node.kwargs.get('enable_gqa', False):
            continue
        kr, vr = _repeat(k), _repeat(v)
        if kr is None or vr is None or kr[1] != vr[1]:
            continue
        key, count, key_snapshot = kr
        value = vr[0]
        if any(min(positions[key_snapshot], positions[vr[2]]) < position < positions[node] for position in effects):
            # Do not replace a copied snapshot with a live view across opaque
            # custom operations that could modify its storage.
            continue
        qs, ks, vs = _shape(q), _shape(key), _shape(value)
        if (any(x is None or len(x) != 4 for x in (qs, ks, vs))
                or not _same(qs[0], ks[0] * count) or not _same(ks[0], vs[0])
                or not _same(qs[1], ks[1]) or not _same(ks[1], vs[1])
                or not _same(ks[2], vs[2]) or not _same(qs[3], ks[3])
                or not all(isinstance(x, int) for x in (qs[1], qs[2], qs[3], vs[3]))):
            continue
        plan = None
        if bias is not None:
            bs = _shape(bias)
            if bs is None or len(bs) != 4 or bs[-2] != 1 or bs[1] not in (1, qs[1]):
                continue
            plan = _bias_plan(bias, count, ks[0])
            if plan is None:
                continue
            if any(positions[snapshot] < position < positions[node]
                   for snapshot in _bias_snapshots(plan) for position in effects):
                continue
        before = set(graph.nodes)
        with graph.inserting_before(node):
            def call(target, *args, **kwargs):
                return graph.call_function(target, args, kwargs)
            def emit(plan):
                if plan[0] in ('base', 'literal'):
                    return plan[1]
                old, children = plan[1:]
                args = [emit(child) for child in children]
                if str(old.target) == 'aten.unsqueeze.default':
                    args += list(old.args[1:])
                return graph.call_function(old.target, tuple(args), dict(old.kwargs))
            batch = call(torch.ops.aten.sym_size.int, key, 0)
            grouped = call(torch.ops.aten.reshape.default, q, [batch, count, qs[1], qs[2], qs[3]])
            grouped = call(torch.ops.aten.permute.default, grouped, [0, 2, 1, 3, 4])
            grouped = call(torch.ops.aten.reshape.default, grouped, [batch, qs[1], count * qs[2], qs[3]])
            compact_bias = emit(plan) if plan is not None else None
            result = call(torch.ops.aten.scaled_dot_product_attention.default,
                grouped, key, value, compact_bias, 0., False,
                **{k: val for k, val in node.kwargs.items() if k not in ('attn_mask', 'dropout_p', 'is_causal')})
            result = call(torch.ops.aten.reshape.default, result, [batch, qs[1], count, qs[2], vs[3]])
            result = call(torch.ops.aten.permute.default, result, [0, 2, 1, 3, 4])
            leading = call(operator.mul, batch, count)
            result = call(torch.ops.aten.reshape.default, result, [leading, qs[1], qs[2], vs[3]])
        result.meta = dict(node.meta)
        for new in (x for x in graph.nodes if x not in before):
            new.meta.setdefault('review_role', 'attention_query_grouping')
            merge_origins(new, node)
        replacements.append({'original_attention': node.name, 'candidate_groups': count,
            'query_length_per_group': qs[2], 'key_source': key.name, 'value_source': value.name,
            'original_query_shape': list(map(str, qs)), 'grouped_query_length': count * qs[2]})
        node.replace_all_uses_with(result)
        graph.erase_node(node)
    cse = _deduplicate(module)
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return {'enabled': True, 'matched': len(replacements), 'replacements': replacements,
            'normalization_cse_nodes': normalized, 'invariant_cse': cse,
            'proof': 'identical K/V and query-independent bias across candidate groups; dropout=0, noncausal SDPA',
            'numeric_policy': 'query regrouping can change fused-kernel selection and floating-point accumulation'}
