# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Static, token-local slicing with a dependency proof for every emitted row.

The same interpreter performs analysis and graph construction. It distributes
static intervals through slices/concatenations and token-local operations, and
slices learned position tensors along their broadcast-aligned token dimension.
Unknown operations and batch mixing fail closed. There is no model-name logic.
"""
from dataclasses import dataclass

import torch
from torch.fx import Node
from torch.fx.node import map_arg
from .review import merge_origins


class ConversionError(ValueError):
    pass


def shape(node):
    value = node.meta.get("val")
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def dependencies(nodes):
    result = {}
    for node in nodes:
        result[node] = ({node} if node.op == "placeholder" else
                        set().union(*(result[x] for x in node.all_input_nodes)))
    return result


POINTWISE = {
    "aten.to.dtype", "aten._to_copy.default", "aten.remainder.Scalar",
    "aten.clamp.default", "aten.div.Tensor", "aten.div.Scalar",
    "aten.mul.Tensor", "aten.mul.Scalar", "aten.add.Tensor", "aten.sub.Tensor",
    "aten.sin.default", "aten.cos.default", "aten.relu.default",
    "aten.sigmoid.default", "aten.tanh.default", "aten.clone.default",
    "aten.contiguous.default", "aten.detach.default",
    "aten.zeros_like.default", "aten.ones_like.default", "aten.full_like.default",
}


@dataclass
class Selected:
    node: object
    deps: set
    # (history input, original token index), independent of feature coordinates.
    origins: set


class TokenSlicer:
    def __init__(self, deps, history, graph=None, env=None):
        self.deps, self.history = deps, history
        self.graph, self.env = graph, {} if env is None else env
        self.memo = {}

    def whole(self, node):
        if self.graph is not None and node not in self.env:
            if node.op == "placeholder":
                raise ConversionError("undeclared_request_dependency: " + node.name)
            self.env[node] = self.graph.node_copy(node, lambda x: self.whole(x).node)
        return Selected(self.env.get(node), self.deps[node], set())

    def emit(self, target, args, kwargs=None):
        values = []
        def unwrap(x):
            if isinstance(x, Selected):
                values.append(x)
                return x.node
            if isinstance(x, tuple):
                return tuple(unwrap(v) for v in x)
            if isinstance(x, list):
                return [unwrap(v) for v in x]
            if isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x
        args, kwargs = unwrap(args), unwrap(kwargs or {})
        output = self.graph.call_function(target, args, kwargs) if self.graph is not None else None
        return Selected(output, set().union(*(x.deps for x in values)),
                        set().union(*(x.origins for x in values)))

    def select(self, node, axis, start, end):
        dims = shape(node)
        if dims is None or not dims:
            raise ConversionError("dynamic_window_shape: tensor dimensions required")
        axis %= len(dims)
        # Batch may be symbolic; token/feature coordinates must remain static.
        # This does not permit slicing or mixing a symbolic batch dimension.
        if (not isinstance(dims[axis], int)
                or any(not isinstance(x, int) for x in dims[1:])):
            raise ConversionError("dynamic_window_shape: static token/feature dimensions required")
        if not 0 <= start < end <= dims[axis]:
            raise ConversionError(f"invalid_token_interval: {node.name}, {axis}, {start}:{end}")
        key = (node, axis, start, end)
        if key not in self.memo:
            self.memo[key] = self._select(node, axis, start, end, dims)
            result = self.memo[key].node
            if isinstance(result, Node) and node.meta.get('review_origins'):
                merge_origins(result, node)
                result.meta['review_slice'] = f'{node.name}: axis {axis}, tokens {start}:{end}'
        return self.memo[key]

    def _select(self, node, axis, start, end, dims):
        if node.op == "placeholder" or not self.deps[node]:
            original = self.whole(node)
            result = self.emit(torch.ops.aten.slice.Tensor, (original, axis, start, end))
            if node in self.history:
                if axis != 1:
                    raise ConversionError("unsupported_history_producer_op: history token axis must be 1")
                result.origins = {(node, p) for p in range(start, end)}
            elif node.op == "placeholder" and axis == 0:
                raise ConversionError("unsupported_history_producer_op: slicing the batch axis")
            return result

        if node.op != "call_function":
            raise ConversionError("unsupported_history_producer_op: " + node.name)
        op = str(node.target)
        args = list(node.args)
        kwargs = dict(node.kwargs)

        if op in POINTWISE:
            def operand(x):
                if not isinstance(x, Node):
                    return x
                xd = shape(x)
                if xd is None:
                    if self.deps[x]:
                        raise ConversionError("unsupported_history_producer_op: dynamic scalar " + x.name)
                    return self.whole(x)
                xa = axis - (len(dims) - len(xd))
                if xa >= 0 and xd[xa] == dims[axis]:
                    return self.select(x, xa, start, end)
                if xa >= 0 and xd[xa] != 1:
                    raise ConversionError("unsupported_history_producer_op: nonstandard token broadcasting")
                # A history-dependent broadcast reduction is not token-local.
                if self.deps[x] & self.history:
                    raise ConversionError("unsupported_history_producer_op: history-dependent broadcast " + x.name)
                return self.whole(x)
            return self.emit(node.target, map_arg(tuple(args), operand), map_arg(kwargs, operand))

        if op == "aten.embedding.default":
            if self.deps[args[0]] or axis >= len(dims) - 1:
                raise ConversionError("unsupported_history_producer_op: dynamic embedding weights or feature slicing")
            args[0] = self.whole(args[0])
            args[1] = self.select(args[1], axis, start, end)
        elif op == "aten.linear.default":
            if axis == len(dims) - 1 or any(self.deps[x] for x in node.all_input_nodes[1:]):
                raise ConversionError("unsupported_history_producer_op: dynamic linear weights or token mixing")
            args[0] = self.select(args[0], axis, start, end)
            args[1:] = [self.whole(x) if isinstance(x, Node) else x for x in args[1:]]
        elif op in {"aten.reshape.default", "aten.view.default"}:
            old = shape(args[0])
            # Supported head split retains B and L verbatim, changing only features.
            if axis != 1 or len(old) < 3 or dims[:2] != old[:2]:
                raise ConversionError("unsupported_history_producer_op: reshape mixes batch or token coordinates")
            args[0] = self.select(args[0], 1, start, end)
            args[1] = [-1, end - start, *dims[2:]]
        elif op == "aten.permute.default":
            perm = args[1]
            if perm[0] != 0:
                raise ConversionError("unsupported_history_producer_op: permuted batch axis")
            args[0] = self.select(args[0], perm[axis], start, end)
        elif op == "aten.unsqueeze.default":
            inserted = args[1] % len(dims)
            if inserted == 0 or inserted == axis:
                raise ConversionError("unsupported_history_producer_op: inserted batch/token axis")
            args[0] = self.select(args[0], axis - int(inserted < axis), start, end)
        elif op == "aten.slice.Tensor":
            source = args[0]
            sd = shape(source)
            sliced_axis = (args[1] if len(args) > 1 else 0) % len(sd)
            lo = args[2] if len(args) > 2 else 0
            hi = args[3] if len(args) > 3 else sd[sliced_axis]
            stride = args[4] if len(args) > 4 else 1
            if not all(isinstance(x, int) for x in (lo, hi, stride)) or stride != 1:
                raise ConversionError("dynamic_or_strided_window: only static unit-stride slices are supported")
            if sliced_axis == 0 and not isinstance(sd[0], int):
                if lo != 0 or hi != 9223372036854775807:
                    raise ConversionError("unsupported_history_producer_op: sliced symbolic batch rows")
                args[0] = self.select(source, axis, start, end)
                return self.emit(node.target, tuple(args), kwargs)
            lo, hi, _ = slice(lo, hi, stride).indices(sd[sliced_axis])
            if sliced_axis == 0 and (lo != 0 or hi != sd[0]):
                raise ConversionError("unsupported_history_producer_op: sliced batch rows")
            if sliced_axis == axis:
                return self.select(source, axis, lo + start, lo + end)
            args[0] = self.select(source, axis, start, end)
        elif op == "aten.cat.default":
            concat_axis = (args[1] if len(args) > 1 else 0) % len(dims)
            if concat_axis == 0:
                raise ConversionError("unsupported_history_producer_op: concatenated batch rows")
            if concat_axis == axis:
                pieces, offset = [], 0
                for part in args[0]:
                    length = shape(part)[axis]
                    lo, hi = max(start, offset), min(end, offset + length)
                    if lo < hi:
                        pieces.append(self.select(part, axis, lo - offset, hi - offset))
                    offset += length
                if len(pieces) == 1:
                    return pieces[0]
                args[0] = pieces
            else:
                args[0] = [self.select(x, axis, start, end) for x in args[0]]
        else:
            raise ConversionError(f"unsupported_history_producer_op: {node.name} ({op}); token/row independence is unproven")
        return self.emit(node.target, tuple(args), kwargs)


def eliminate_common_subexpressions(graph):
    """Hoist identical pure expressions, including request-invariant K/V.

Conservative allowlist excludes allocation, randomness and mutation. No name or
business-semantic heuristic determines which branches are merged.
"""
    pure = POINTWISE - {"aten.clone.default", "aten.contiguous.default"}
    pure |= {"aten.embedding.default", "aten.linear.default", "aten.layer_norm.default",
             "aten.reshape.default", "aten.view.default", "aten.permute.default",
             "aten.unsqueeze.default", "aten.slice.Tensor", "aten.select.int",
             "aten.cat.default", "aten.stack.default", "aten.constant_pad_nd.default"}
    def freeze(x):
        if isinstance(x, (tuple, list)):
            return tuple(freeze(v) for v in x)
        if isinstance(x, dict):
            return tuple((k, freeze(v)) for k, v in sorted(x.items()))
        return x
    seen, count = {}, 0
    for node in list(graph.nodes):
        # Export can chain redundant .to(dtype) calls across unrolled steps.
        # Removing only true no-op casts lets ordinary CSE see shared features;
        # precision-changing casts (including BF16 rounding) are retained.
        if node.op == 'call_function' and str(node.target) == 'aten.to.dtype':
            source = node.args[0]
            val = source.meta.get('val')
            dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get('dtype')
            copy = node.args[3] if len(node.args) > 3 else node.kwargs.get('copy', False)
            memory_format = node.args[4] if len(node.args) > 4 else node.kwargs.get('memory_format')
            if isinstance(val, torch.Tensor) and val.dtype == dtype and not copy and memory_format is None:
                merge_origins(source, node)
                node.replace_all_uses_with(source)
                graph.erase_node(node)
                count += 1
                continue
        # These checks concern static tensor metadata, not model data. Prove
        # them against export metadata before removing them; otherwise their
        # side-effect marker keeps dead history computation alive after slicing.
        if node.op == 'call_function' and str(node.target) == 'aten._assert_tensor_metadata.default':
            val = node.args[0].meta.get('val')
            metadata = dict(node.kwargs)
            static_only = len(node.args) == 1
            if 3 <= len(node.args) <= 4 and node.args[1:3] == (None, None):
                static_only = True
                if len(node.args) == 4:
                    metadata['dtype'] = node.args[3]
            if (static_only and not node.users and isinstance(val, torch.Tensor)
                    and set(metadata) <= {'dtype', 'device', 'layout'}
                    and all(getattr(val, k) == v for k, v in metadata.items())):
                graph.erase_node(node)
                count += 1
                continue
        if node.op != "get_attr" and not (node.op == "call_function" and str(node.target) in pure):
            continue
        key = (node.op, node.target, freeze(node.args), freeze(node.kwargs))
        if key in seen:
            merge_origins(seen[key], node)
            node.replace_all_uses_with(seen[key])
            graph.erase_node(node)
            count += 1
        else:
            seen[key] = node
    return count
