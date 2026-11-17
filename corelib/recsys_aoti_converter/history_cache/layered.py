# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Mask-proven history reuse through repeated stacks of fused-UVQK attention.

Matches ATen dataflow, not model class/module/node names. The initial support
is a fixed history suffix and inference-only SiLU HSTU attention. Every matched
layer retains its original arithmetic; token dimensions and K/V sources change.
"""
from dataclasses import dataclass, asdict
import operator

import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils import _pytree as pytree

from .slicing import ConversionError, Selected, TokenSlicer, POINTWISE, shape, eliminate_common_subexpressions
from .context_guard import UniformSlice, UNKNOWN

PREPROCESS_POINTWISE = POINTWISE | {
    'aten.alias.default', 'aten.log.default', 'aten.log1p.default',
    'aten.floor.default', 'aten.ceil.default', 'aten.round.default', 'aten.abs.default',
    'aten.gelu.default', 'aten.silu.default', 'aten.where.self',
    'aten.masked_fill.Scalar', 'aten.eq.Scalar', 'aten.eq.Tensor',
    'aten.ne.Scalar', 'aten.logical_and.default', 'aten.bitwise_not.default',
    'aten.isnan.default', 'aten.isinf.default', 'aten.__and__.Tensor',
    'aten.pow.Tensor_Scalar', 'aten.remainder.Tensor'}


class HistorySlicer(TokenSlicer):
    """Distribute a fixed token slice through row-local preprocessing.

    This eliminates candidate dependencies before history embedding, even when
    the original preprocessing first concatenates candidate/history fields.
    Reductions or normalization across the selected token axis are rejected.
    """
    def __init__(self, deps, graph=None, env=None, whole_copy=None, batch=None):
        super().__init__(deps, set(), graph, env)
        self.whole_copy, self.batch = whole_copy, batch

    def whole(self, node):
        if self.whole_copy is None:
            return super().whole(node)
        return Selected(self.whole_copy(node), self.deps[node], set())

    def unstack_select(self, source, index, axis, start, end):
        if op(source, {'aten.stack.default'}) and (len(source.args) == 1 or source.args[1] == 0):
            return self.select(source.args[0][index], axis, start, end)
        if op(source, {'aten.pad.default'}) and all(x == 0 for x in source.args[1]):
            return self.unstack_select(source.args[0], index, axis, start, end)
        if op(source, {'aten.bucketize.Tensor'}) and not self.deps[source.args[1]]:
            selected = self.unstack_select(source.args[0], index, axis, start, end)
            return self.emit(source.target, (selected, self.whole(source.args[1]), *source.args[2:]), dict(source.kwargs))
        allowed = PREPROCESS_POINTWISE
        if op(source, allowed):
            dims = shape(source)
            def operand(x):
                if not isinstance(x, Node): return x
                xd = shape(x)
                if xd is not None and len(xd) == len(dims) and xd[0] == dims[0]:
                    return self.unstack_select(x, index, axis, start, end)
                return self.whole(x)
            return self.emit(source.target, torch.fx.node.map_arg(source.args, operand),
                             torch.fx.node.map_arg(source.kwargs, operand))
        raise ConversionError(f'unproven feature-stack preprocessing: {source.name} ({source.target})')

    def _select(self, node, axis, start, end, dims):
        target = str(node.target)
        args, kwargs = list(node.args), dict(node.kwargs)
        if target == 'aten.bucketize.Tensor' and not self.deps[args[1]]:
            return self.emit(node.target, (self.select(args[0], axis, start, end), self.whole(args[1]), *args[2:]), kwargs)
        if node.target is operator.getitem and op(args[0], {'aten.unbind.int'}):
            unbind, index = args
            if (len(unbind.args) == 1 or unbind.args[1] == 0) and isinstance(index, int):
                source, unary = unbind.args[0], []
                allowed = PREPROCESS_POINTWISE
                while (op(source, allowed) and source.args and isinstance(source.args[0], Node)
                       and len(source.all_input_nodes) == 1):
                    unary.append(source)
                    source = source.args[0]
                if op(source, {'aten.stack.default'}) and (len(source.args) == 1 or source.args[1] == 0):
                    selected = self.select(source.args[0][index], axis, start, end)
                    for operation in reversed(unary):
                        selected = self.emit(operation.target, (selected, *operation.args[1:]), dict(operation.kwargs))
                    return selected
                return self.unstack_select(unbind.args[0], index, axis, start, end)
        if target == 'aten.slice.Tensor':
            source = args[0]
            old = shape(source)
            sliced_axis = (args[1] if len(args) > 1 else 0) % len(old)
            lo = args[2] if len(args) > 2 and args[2] is not None else 0
            hi = args[3] if len(args) > 3 and args[3] is not None else 9223372036854775807
            step = args[4] if len(args) > 4 else 1
            if step != 1:
                raise ConversionError('history preprocessing has a strided slice')
            if sliced_axis == axis:
                lo, hi, _ = slice(lo, hi).indices(old[axis])
                return self.select(source, axis, start + lo, end + lo)
            chosen = self.select(source, axis, start, end)
            return self.emit(node.target, (chosen, sliced_axis, lo, hi, step), kwargs)
        if target == 'aten.layer_norm.default':
            if axis >= len(dims) - len(args[1]):
                raise ConversionError('history normalization mixes selected tokens')
            args[0] = self.select(args[0], axis, start, end)
            args[1:] = [self.whole(x) if isinstance(x, Node) else x for x in args[1:]]
            return self.emit(node.target, tuple(args), kwargs)
        if target in {'aten.sum.dim_IntList', 'aten.mean.dim'}:
            old = shape(args[0])
            reduced = {d % len(old) for d in args[1]}
            kept = len(args) > 2 and args[2]
            source_axis = axis if kept else [i for i in range(len(old)) if i not in reduced][axis]
            if source_axis in reduced or 0 in reduced:
                raise ConversionError('history reduction mixes selected tokens/batch')
            args[0] = self.select(args[0], source_axis, start, end)
            return self.emit(node.target, tuple(args), kwargs)
        if target == 'aten.stack.default':
            inserted = (args[1] if len(args) > 1 else 0) % len(dims)
            if inserted == 0:
                raise ConversionError('history preprocessing has a leading feature stack')
            if inserted == axis:
                selected = [self.whole(x) for x in args[0][start:end]]
            else:
                selected = [self.select(x, axis-int(inserted < axis), start, end) for x in args[0]]
            return self.emit(node.target, (selected, inserted), kwargs)
        if target in {'aten.reshape.default', 'aten.view.default'}:
            old = shape(args[0])
            if axis == 1 and len(old) >= 3 and old[:2] == dims[:2]:
                source = self.select(args[0], 1, start, end)
                return self.emit(node.target, (source, [self.batch, end-start, *dims[2:]]))
            if start == 0 and end == dims[axis] and dims[0] == old[0]:
                return self.whole(node)
        pointwise = PREPROCESS_POINTWISE
        if target in pointwise:
            def operand(x):
                if not isinstance(x, Node): return x
                xd = shape(x)
                if xd is None: return self.whole(x)
                xa = axis - (len(dims)-len(xd))
                if xa >= 0 and xd[xa] == dims[axis]:
                    return self.select(x, xa, start, end)
                return self.whole(x)
            return self.emit(node.target, torch.fx.node.map_arg(tuple(args), operand),
                             torch.fx.node.map_arg(kwargs, operand))
        return super()._select(node, axis, start, end, dims)


def op(n, names):
    return isinstance(n, Node) and n.op == 'call_function' and str(n.target) in names


def unwrap(n):
    while isinstance(n, Node):
        if op(n, {'aten.clone.default', 'aten.contiguous.default', 'aten.detach.default'}):
            n = n.args[0]
        elif op(n, {'aten.dropout.default'}) and n.args[2] is False:
            n = n.args[0]
        elif op(n, {'aten.to.dtype', 'aten._to_copy.default'}):
            a, b = n.args[0].meta.get('val'), n.meta.get('val')
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device:
                n = n.args[0]
            else:
                break
        else:
            break
    return n


def ancestors(*nodes):
    result, todo = set(), list(nodes)
    while todo:
        n = todo.pop()
        if not isinstance(n, Node) or n in result:
            continue
        result.add(n)
        todo.extend(n.all_input_nodes)
    return result


def value_dependencies(nodes):
    result = {}
    for n in nodes:
        if n.op == 'placeholder':
            result[n] = {n}
        elif op(n, {'aten.sym_size.int'}) and n.args[1] == 0:
            result[n] = set()  # common batch is an ABI shape, not a feature
        else:
            result[n] = set().union(*(result[x] for x in n.all_input_nodes))
    return result


def attr(module, target):
    value = module
    for piece in target.split('.'):
        value = getattr(value, piece)
    return value


class Expressions:
    def __init__(self, module):
        self.module, self.memo = module, {}

    def key(self, value):
        if isinstance(value, Node):
            n = unwrap(value)
            if n not in self.memo:
                if n.op == 'get_attr':
                    tensor = attr(self.module, n.target)
                    if isinstance(tensor, torch.Tensor) and tensor.numel() <= 16:
                        self.memo[n] = ('constant', str(tensor.dtype), tuple(tensor.shape), tuple(tensor.flatten().tolist()))
                    else:
                        self.memo[n] = ('attribute', n.target)
                elif n.op == 'placeholder':
                    self.memo[n] = ('input', n.name)
                else:
                    self.memo[n] = (str(n.target), self.key(n.args), self.key(n.kwargs))
            return self.memo[n]
        if isinstance(value, (list, tuple)):
            return tuple(self.key(x) for x in value)
        if isinstance(value, dict):
            return tuple((k, self.key(v)) for k, v in sorted(value.items()))
        return value


def head_projection(node, index):
    head = unwrap(node)
    if not op(head, {'aten.view.default', 'aten.reshape.default'}):
        return None
    selected = unwrap(head.args[0])
    if not (selected.op == 'call_function' and selected.target is operator.getitem and selected.args[1] == index):
        return None
    split = selected.args[0]
    if not op(split, {'aten.split_with_sizes.default'}) or split.args[2] != -1:
        return None
    linear = unwrap(split.args[0])
    if not op(linear, {'aten.linear.default'}):
        return None
    dims = shape(node)
    if not dims or len(dims) != 4 or list(split.args[1]) != [dims[2] * dims[3]] * 4:
        return None
    return linear, split


@dataclass
class Layer:
    output: Node
    x: Node
    key: Node
    value: Node
    projection: Node
    denominator: Node
    mask: Node
    norm1: Node
    norm2: Node
    linear: Node
    internal: set


def match_layer(output):
    if not op(output, {'aten.add.Tensor'}) or len(output.args) != 2:
        return None
    projected, residual = output.args
    linear = unwrap(projected)
    if not op(linear, {'aten.linear.default'}):
        projected, residual = residual, projected
        linear = unwrap(projected)
    if not op(linear, {'aten.linear.default'}):
        return None
    reshape = unwrap(linear.args[0])
    if not op(reshape, {'aten.reshape.default', 'aten.view.default'}):
        return None
    trans = unwrap(reshape.args[0])
    if not op(trans, {'aten.transpose.int'}) or tuple(trans.args[1:]) != (1, 2):
        return None
    gate = unwrap(trans.args[0])
    if not op(gate, {'aten.mul.Tensor'}):
        return None
    u, norm2 = gate.args
    norm2 = unwrap(norm2)
    if not op(norm2, {'aten.layer_norm.default'}):
        return None
    av = unwrap(norm2.args[0])
    if not op(av, {'aten.einsum.default'}) or av.args[0] != 'bhqk,bkhv->bhqv':
        return None
    probabilities, value = av.args[1]
    probabilities = unwrap(probabilities)
    if not op(probabilities, {'aten.mul.Tensor'}):
        return None
    divided, mask_head = probabilities.args
    if not op(divided, {'aten.div.Tensor'}) or not op(mask_head, {'aten.unsqueeze.default'}) or mask_head.args[1] != 1:
        return None
    activated, denominator = divided.args
    if not op(activated, {'aten.silu.default'}):
        return None
    scaled = activated.args[0]
    if not op(scaled, {'aten.mul.Tensor', 'aten.div.Tensor'}) or not isinstance(scaled.args[1], (int, float)):
        return None
    qk = scaled.args[0]
    if not op(qk, {'aten.einsum.default'}) or qk.args[0] != 'bqhd,bkhd->bhqk':
        return None
    query, key = qk.args[1]
    branches = [head_projection(n, i) for n, i in ((value, 1), (query, 2), (key, 3))]
    if any(x is None for x in branches) or len({x[0] for x in branches}) != 1:
        return None
    projection = branches[0][0]
    norm1 = unwrap(projection.args[0])
    if not op(norm1, {'aten.layer_norm.default'}) or unwrap(norm1.args[0]) is not unwrap(residual):
        return None
    u = unwrap(u)
    if not op(u, {'aten.transpose.int'}) or tuple(u.args[1:]) != (1, 2):
        return None
    u = unwrap(u.args[0])
    if not op(u, {'aten.silu.default'}) or head_projection(u.args[0], 0) != branches[0]:
        return None
    x, mask = unwrap(residual), mask_head.args[0]
    # Stop traversal at boundaries, rather than subtracting all their ancestors:
    # a mask unsqueeze shared with an earlier layer is also an ancestor of x,
    # but still needs the current history/candidate query slice here.
    internal, todo = set(), [output]
    boundaries = {x, denominator, mask}
    while todo:
        n = todo.pop()
        if (n in boundaries or n in internal or n.op in {'placeholder', 'get_attr'}
                or op(n, {'aten.sym_size.int'})):
            continue
        internal.add(n)
        todo.extend(n.all_input_nodes)
    # The supplied export carries dtype/device assertions with no size or
    # stride checks. The re-export and dtype audit validate these contracts;
    # retaining the old assertion inputs would resurrect the full-token path.
    assertions = {u for n in internal for u in n.users
                  if op(u, {'aten._assert_tensor_metadata.default'})
                  and u.args[1:3] == (None, None) and not u.users}
    internal.update(assertions)
    return Layer(output, x, key, value, projection, denominator, mask, norm1, norm2, linear, internal)


def prove_mask(module, layer, start, length, deps, expressions):
    """Prove static zero H→C edges for every supported batch; retain validity."""
    mask = unwrap(layer.mask)
    if not op(mask, {'aten.mul.Tensor'}):
        raise ConversionError('layered_mask: expected static mask times query/key validity')
    constant, outer = mask.args
    if deps[constant]:
        constant, outer = outer, constant
    if deps[constant] or not op(outer, {'aten.matmul.default', 'aten.bmm.default'}):
        raise ConversionError('layered_mask: cannot separate a static mask from validity')
    left, right = outer.args
    if not (op(left, {'aten.unsqueeze.default'}) and left.args[1] == -1
            and op(right, {'aten.unsqueeze.default'}) and right.args[1] == 1
            and expressions.key(left.args[0]) == expressions.key(right.args[0])):
        raise ConversionError('layered_mask: expected outer product of one validity vector')
    denom = unwrap(layer.denominator)
    if not op(denom, {'aten.view.default', 'aten.reshape.default'}):
        raise ConversionError('layered_normalization: missing per-request valid count')
    clamp = denom.args[0]
    if not op(clamp, {'aten.clamp.default'}) or tuple(clamp.args[1:]) != (1,):
        raise ConversionError('layered_normalization: expected count clamped to one')
    total = clamp.args[0]
    if not op(total, {'aten.sum.dim_IntList'}) or list(total.args[1]) != [1] or expressions.key(total.args[0]) != expressions.key(left.args[0]):
        raise ConversionError('layered_normalization: denominator must count the original validity vector')
    allowed = {'aten.zeros.default', 'aten.ones.default', 'aten.full.default', 'aten.eye.default',
        'aten.arange.default', 'aten.arange.start', 'aten.arange.start_step', 'aten.unsqueeze.default',
        'aten.slice.Tensor', 'aten.slice_scatter.default', 'aten.select_scatter.default',
        'aten.copy.default', 'aten.fill.Tensor', 'aten.fill.Scalar', 'aten.lift_fresh_copy.default',
        'aten.clone.default', 'aten.repeat.default', 'aten.expand.default', 'aten._to_copy.default',
        'aten.to.dtype', 'aten.le.Tensor', 'aten.ge.Tensor', 'aten.lt.Tensor', 'aten.gt.Tensor',
        'aten.add.Tensor', 'aten.mul.Tensor', 'aten.view.default', 'aten.reshape.default',
        'aten.sym_size.int', 'aten.tril.default', 'aten.triu.default'}
    for batch in range(1, 9):
        memo = {}
        def evaluate(n):
            if n in memo:
                return memo[n]
            if n.op == 'get_attr':
                value = attr(module, n.target)
                if isinstance(value, torch.Tensor) and value.numel() > (start + length) ** 2:
                    raise ConversionError('layered_mask: oversized constant')
            elif op(n, {'aten.sym_size.int'}) and n.args[1] == 0:
                value = batch
            elif n.op == 'call_function' and str(n.target) in allowed:
                value = n.target(*torch.fx.node.map_arg(n.args, evaluate), **torch.fx.node.map_arg(n.kwargs, evaluate))
            else:
                raise ConversionError(f'layered_mask: unproven constant operation {n.target}')
            memo[n] = value
            return value
        with torch.no_grad():
            static = evaluate(constant)
        if tuple(static.shape) != (batch, start + length, start + length) or not torch.isfinite(static).all() or torch.count_nonzero(static[:, start:, :start]):
            raise ConversionError('layered_mask: history may attend to candidates')
    return constant, left.args[0]


@dataclass
class LayeredPlan:
    module: GraphModule
    layers: list
    chains: list
    roots: list
    deps: dict
    producer_input_indices: list
    report: dict
    validity: Node

    @property
    def banks(self):
        return len(self.chains[0])


def prove_history_only_context(validity, spec, deps, inputs, producer_indices):
    """Elide a defensive check only when context follows the snapshot contract.

    prove_mask has already established that normalization is the clamped sum
    of this exact validity vector. A fixed candidate prefix plus a history-only
    validity suffix therefore makes the entire checked context history-only.
    """
    proof = {'elided': False, 'candidate_length': spec.history_start,
             'history_length': spec.length, 'policy': spec.policy}
    candidate = UniformSlice()(validity, 1, 0, spec.history_start)
    if candidate is UNKNOWN or candidate not in (False, True):
        proof['reason'] = 'non-history validity is not statically constant boolean'
        return proof
    proof['candidate_valid_value'] = bool(candidate)
    declared = spec.history_input_indices or producer_indices
    allowed = {inputs[i] for i in declared}
    try:
        selected = HistorySlicer(deps).select(
            validity, 1, spec.history_start, spec.history_start + spec.length)
    except ConversionError as error:
        proof['reason'] = 'history validity dependency is unproven: ' + str(error)
        return proof
    proof['history_validity_input_indices'] = [i for i, n in enumerate(inputs) if n in selected.deps]
    proof['immutable_input_indices'] = list(declared)
    if not selected.deps <= allowed:
        proof['reason'] = 'history validity depends on undeclared immutable inputs'
        return proof
    proof.update(elided=True,
                 reason='constant non-history validity; history validity depends only on immutable inputs',
                 normalization='clamp(sum(original validity), min=1)',
                 reuse_contract='snapshot_id identifies unchanged history values and validity')
    return proof


def analyze_layered(exported, spec):
    if spec.history_start == 0:
        return None
    if spec.policy != 'immutable_snapshot' or spec.history_start < 1:
        raise ConversionError('layered history requires an immutable fixed history suffix')
    module = exported.module()
    eliminate_common_subexpressions(module.graph)
    module.graph.eliminate_dead_code()
    module.recompile()
    nodes = list(module.graph.nodes)
    deps, expressions = value_dependencies(nodes), Expressions(module)
    layers = [x for n in nodes if (x := match_layer(n)) is not None]
    if not layers:
        raise ConversionError('layered_attention_missing: no supported fused-UVQK/SiLU layers')
    by_output = {x.output: x for x in layers}
    chains, roots = [], []
    for layer in layers:
        dims = shape(layer.key)
        if dims[1:] != (spec.history_start + spec.length, spec.heads, spec.head_dim):
            raise ConversionError('layered geometry differs from CacheSpec')
        if layer.x in by_output:
            parent = by_output[layer.x]
            chain = next((x for x in chains if x[-1] is parent), None)
            if chain is None:
                raise ConversionError('branched or reordered attention stack is unsupported')
            chain.append(layer)
        else:
            root = unwrap(layer.x)
            if not op(root, {'aten.cat.default'}) or len(root.args[0]) != 2 or root.args[1] != 1:
                raise ConversionError('layered root must concatenate candidate and history tokens')
            c, h = root.args[0]
            if shape(c)[1] != spec.history_start or shape(h)[1] != spec.length:
                raise ConversionError('declared fixed history suffix differs from token concatenation')
            roots.append((root, c, h))
            chains.append([layer])
    if len({len(c) for c in chains}) != 1:
        raise ConversionError('repeated attention stacks have different depths')
    base = chains[0]
    constant, validity = prove_mask(module, base[0], spec.history_start, spec.length, deps, expressions)
    def parameters(layer):
        return expressions.key((layer.norm1.args[1:], layer.norm2.args[1:], layer.projection.args[1:], layer.linear.args[1:]))
    for index, chain in enumerate(chains):
        if expressions.key(roots[index][2]) != expressions.key(roots[0][2]):
            raise ConversionError('repeated stacks do not share an identical history expression')
        for depth, layer in enumerate(chain):
            if (parameters(layer) != parameters(base[depth])
                    or expressions.key(layer.mask) != expressions.key(base[0].mask)
                    or expressions.key(layer.denominator) != expressions.key(base[0].denominator)):
                raise ConversionError('repeated history layer parameters/mask/normalization differ')
    inputs = [n for n in nodes if n.op == 'placeholder']
    selected_history = HistorySlicer(deps).select(roots[0][2], 1, 0, spec.length)
    required = selected_history.deps
    indices = [i for i, n in enumerate(inputs) if n in required]
    if spec.history_input_indices and not required <= {inputs[i] for i in spec.history_input_indices}:
        raise ConversionError('history producer depends on undeclared reusable inputs')
    # Intermediate tensors may only feed their matched layer. Weights/shape
    # constants are shared freely. Reject unaccounted feature consumers.
    all_internal = set().union(*(layer.internal for layer in layers))
    for layer in layers:
        for n in layer.internal - {layer.output}:
            if deps[n] and any(u not in all_internal for u in n.users):
                raise ConversionError(f'layered intermediate has an external consumer: {n.name}')
    banks = [{'layer': i, 'key_node': layer.key.name, 'value_node': layer.value.name,
        'prefix_length': spec.length, 'request_tail_length': 0, 'storage_length': spec.length,
        'position_range': [spec.history_start, spec.history_start + spec.length],
        'history_source_offsets': [0]} for i, layer in enumerate(base)]
    report = {'spec': asdict(spec), 'pattern': 'mask-proven repeated fused-UVQK SiLU attention stacks',
        'banks': banks, 'bank_count': len(banks), 'layer_count': len(banks),
        'stack_passes': len(chains), 'producer_inputs': indices, 'history_value_input_indices': indices,
        'history_start': spec.history_start, 'cached_prefix_rows': spec.length * len(banks),
        'history_length': spec.length, 'context_guard': 'history validity and original total valid-token count',
        'original_nodes': len(nodes), 'static_mask_node': constant.name,
        'storage': 'one fixed-length history slot per snapshot, with one K/V pair per layer; no paged attention'}
    context_proof = prove_history_only_context(validity, spec, deps, inputs, indices)
    report['context_guard_elision'] = context_proof
    if context_proof['elided']:
        report['context_guard'] = None
    return LayeredPlan(module, layers, chains, roots, deps, indices, report, validity)


def rewrite_layered(plan, spec, *, sdpa=False, review_dir=None, source_modules=None):
    if sdpa:
        raise ConversionError('SiLU HSTU attention is not softmax SDPA; omit --sdpa')
    from .review import tag_original, mark_nodes, write_review
    module, graph, env = plan.module, Graph(), {}
    if review_dir is not None:
        tag_original(module)
    original = list(module.graph.nodes)
    inputs = [n for n in original if n.op == 'placeholder']
    ids = graph.placeholder('snapshot_ids')
    enabled = graph.placeholder('enable_offload')
    pools = [graph.placeholder(f'pool_{i}') for i in range(plan.banks)]
    for n in inputs:
        env[n] = graph.placeholder(n.name)
        env[n].meta = dict(n.meta)
    def copy(n):
        if n not in env:
            env[n] = graph.node_copy(n, copy)
        return env[n]
    def call(target, *args):
        return graph.call_function(target, args)
    def sliced(n, axis, start, end):
        return call(torch.ops.aten.slice.Tensor, n, axis, start, end)
    first = plan.chains[0][0]
    denom, mask = copy(first.denominator), copy(first.mask)
    context = None
    if plan.report.get('context_guard'):
        history_validity = sliced(copy(plan.validity), 1, spec.history_start, spec.history_start + spec.length)
        count = call(torch.ops.aten.reshape.default, denom, [-1, 1])
        context = call(torch.ops.aten.cat.default, [count, history_validity], 1)
    prepare = call(torch.ops.history_single.prepare.default, ids, env[inputs[0]], spec.length, plan.banks, context)
    ticket, rows, slots = [call(operator.getitem, prepare, i) for i in range(3)]
    raw = [env[inputs[i]] for i in plan.producer_input_indices]
    packed = call(torch.ops.history_single.pack.default, [*raw, denom, mask], rows, slots, ticket)
    penv = {inputs[i]: call(operator.getitem, packed, j) for j, i in enumerate(plan.producer_input_indices)}
    pdenom = call(operator.getitem, packed, len(raw))
    pmask = call(operator.getitem, packed, len(raw) + 1)
    missing = call(operator.getitem, packed, len(raw) + 2)
    pmask = sliced(sliced(pmask, 1, spec.history_start, spec.history_start + spec.length), 2, spec.history_start, spec.history_start + spec.length)
    def producer_copy(n):
        if n in penv:
            return penv[n]
        if n.op == 'get_attr':
            result = copy(n)
        elif op(n, {'aten.sym_size.int'}) and n.args[1] == 0:
            result = call(torch.ops.aten.sym_size.int, next(iter(penv.values())), 0)
        elif n.op == 'placeholder':
            raise ConversionError('undeclared history producer value: ' + n.name)
        else:
            # Avoid ambiguous reshape [0,...,-1] on an all-hit (zero-miss) batch.
            dims = shape(n)
            if op(n, {'aten.view.default', 'aten.reshape.default'}) and dims and len(dims) > 1 and all(isinstance(d, int) for d in dims[1:]):
                batch = call(torch.ops.aten.sym_size.int, next(iter(penv.values())), 0)
                source_args = (producer_copy(n.args[0]), [batch, *dims[1:]])
            else:
                source_args = torch.fx.node.map_arg(n.args, producer_copy)
            result = graph.create_node(n.op, n.target, source_args, torch.fx.node.map_arg(n.kwargs, producer_copy))
            result.meta = dict(n.meta)
        penv[n] = result
        return result
    producer_batch = Selected(call(torch.ops.aten.sym_size.int, next(iter(penv.values())), 0), set(), set())
    history = HistorySlicer(plan.deps, graph, penv, producer_copy, producer_batch).select(
        plan.roots[0][2], 1, 0, spec.length).node
    history_states = {0: history}
    gathered, insertions = {}, []

    def clone_layer(layer, x, mask_value, denominator, length):
        local = {layer.x: x, layer.mask: mask_value, layer.denominator: denominator}
        local[unwrap(layer.x)] = x
        def emit(n):
            if n in local:
                return local[n]
            if n not in layer.internal:
                return copy(n)
            dims = shape(n)
            if op(n, {'aten.view.default', 'aten.reshape.default'}) and dims and len(dims) in (3, 4) and dims[1] == spec.history_start + spec.length:
                batch = call(torch.ops.aten.sym_size.int, x, 0)
                source_args = (emit(n.args[0]), [batch, length, *dims[2:]])
            else:
                source_args = torch.fx.node.map_arg(n.args, emit)
            result = graph.create_node(n.op, n.target, source_args, torch.fx.node.map_arg(n.kwargs, emit))
            result.meta = dict(n.meta)
            local[n] = result
            return result
        return local, emit

    outputs = {l.output: (p, d, l) for p, chain in enumerate(plan.chains) for d, l in enumerate(chain)}
    internal = set().union(*(l.internal for l in plan.layers))
    roots = {root: c for root, c, h in plan.roots}
    for n in original:
        if n.op == 'placeholder':
            continue
        if n in roots:
            env[n] = copy(roots[n])
        elif n in outputs:
            pass_index, depth, layer = outputs[n]
            before = set(graph.nodes)
            if depth not in gathered:
                base = plan.chains[0][depth]
                local, emit = clone_layer(base, history_states[depth], pmask, pdenom, spec.length)
                k, v = emit(base.key), emit(base.value)
                canonical = [call(torch.ops.aten.permute.default, t, [0, 2, 1, 3]) for t in (k, v)]
                filled = call(torch.ops.history_single.fill.default, pools[depth], missing, *canonical, ticket, depth)
                history_states[depth + 1] = emit(base.output)
                ready = call(torch.ops.history_single.onboard_wait.default, ticket, filled, copy(layer.x))
                dense = call(torch.ops.history_single.gather.default, pools[depth], slots, ready, depth)
                gathered[depth] = [call(torch.ops.aten.permute.default, call(operator.getitem, dense, j), [0, 2, 1, 3]) for j in range(2)]
                insertions.append({'bank': depth, 'layer': depth, 'fill': filled.name, 'wait': ready.name, 'gather': dense.name})
            cmask = sliced(copy(layer.mask), 1, 0, spec.history_start)
            local, emit = clone_layer(layer, copy(layer.x), cmask, copy(layer.denominator), spec.history_start)
            candidate = [emit(layer.key), emit(layer.value)]
            for old, current, cached in zip((layer.key, layer.value), candidate, gathered[depth]):
                local[old] = call(torch.ops.aten.cat.default, [current, cached], 1)
            env[n] = emit(layer.output)
            if review_dir is not None:
                mark_nodes([x for x in graph.nodes if x not in before], f'layer_{depth:02d}_pass_{pass_index:02d}', depth)
        elif n.op == 'output':
            values = [copy(x) for x in pytree.tree_leaves(n.args[0])]
            final = call(torch.ops.history_single.offload_launch.default, ticket, values[-1], enabled)
            graph.output(tuple([*values, final]))
        elif n not in internal and n not in env:
            copy(n)
    converted = GraphModule(module, graph, 'LayeredHistoryModel')
    eliminate_common_subexpressions(graph)
    graph.eliminate_dead_code()
    graph.lint()
    converted.recompile()
    report = dict(plan.report, insertion_sites=insertions, execution='one end-to-end ExportedProgram',
        attention={'enabled': False, 'matched': 0, 'kind': 'original SiLU HSTU'},
        native_fill_gather_fused=False, copied_model_sources=False,
        output_abi='original flattened outputs plus private submission ticket',
        producer_policy='history-only missing rows, interleaved at the original layer boundaries',
        consumer_policy='candidate-only rows; layer history K/V gathered once and reused across passes')
    if review_dir is not None:
        report['review'] = write_review(module, converted, report, review_dir, source_modules)
    return converted, report
