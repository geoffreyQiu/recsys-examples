# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Conservative scalar-value proof for a static token interval of an FX tensor.

This reads ATen structure and tensor metadata, never sample values. Unknown
operations remain unknown. It is intentionally narrower than constant folding:
only constructors, simple shape operations and boolean validity expressions are
needed to show that a non-history validity prefix is invariant.
"""
import operator

import torch
from torch.fx import Node

from .slicing import shape

UNKNOWN = object()


class UniformSlice:
    def __init__(self):
        self.memo = {}

    def __call__(self, node, axis, start, end):
        if not isinstance(node, Node):
            return node if isinstance(node, (bool, int, float)) else UNKNOWN
        key = (node, axis, start, end)
        if key not in self.memo:
            try:
                value = self._value(node, axis, start, end)
                # Shape operations such as cat/stack may promote dtypes too.
                self.memo[key] = self.typed(node, value) if value is not UNKNOWN else UNKNOWN
            except (TypeError, ValueError, IndexError, RuntimeError, OverflowError):
                self.memo[key] = UNKNOWN
        return self.memo[key]

    @staticmethod
    def typed(node, value):
        """Honor constructor/cast dtype, including rounding and integer casts."""
        meta = node.meta.get('val')
        if not isinstance(meta, torch.Tensor):
            return UNKNOWN
        return torch.tensor(value, dtype=meta.dtype, device='cpu').item()

    def operand(self, operand, output, axis, start, end):
        if not isinstance(operand, Node):
            return self(operand, axis, start, end)
        old, new = shape(operand), shape(output)
        if old is None or new is None:
            return UNKNOWN
        mapped_axis = axis - (len(new) - len(old))
        if not old:
            return self(operand, 0, 0, 1)
        if mapped_axis < 0:
            # The operand broadcasts along the selected axis, but may vary in
            # a feature axis. A single element would not establish uniformity.
            return UNKNOWN
        if old[mapped_axis] == 1:
            return self(operand, mapped_axis, 0, 1)
        if old[mapped_axis] != new[axis]:
            return UNKNOWN
        return self(operand, mapped_axis, start, end)

    def _value(self, n, axis, start, end):
        if n.op != 'call_function':
            return UNKNOWN
        dims = shape(n)
        if dims is None:
            return UNKNOWN
        if dims:
            axis %= len(dims)
            if not isinstance(dims[axis], int) or not 0 <= start < end <= dims[axis]:
                return UNKNOWN
        target, args = str(n.target), n.args
        if target in {'aten.ones.default', 'aten.ones_like.default'}:
            return self.typed(n, 1)
        if target in {'aten.zeros.default', 'aten.zeros_like.default'}:
            return self.typed(n, 0)
        if target in {'aten.full.default', 'aten.full_like.default', 'aten.new_full.default'}:
            index = 2 if target == 'aten.new_full.default' else 1
            fill = args[index] if len(args) > index else n.kwargs.get('fill_value', UNKNOWN)
            return self.typed(n, fill) if isinstance(fill, (bool, int, float)) else UNKNOWN
        if target in {'aten.alias.default', 'aten.clone.default', 'aten.detach.default',
                      'aten.contiguous.default'}:
            return self(args[0], axis, start, end)
        if target in {'aten.to.dtype', 'aten._to_copy.default'}:
            value = self(args[0], axis, start, end)
            return self.typed(n, value) if value is not UNKNOWN else UNKNOWN
        if target == 'aten.slice.Tensor':
            old = shape(args[0])
            if old is None:
                return UNKNOWN
            dim = (args[1] if len(args) > 1 else 0) % len(old)
            lo, hi = args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            if step != 1:
                return UNKNOWN
            if dim != axis:
                # Proving the entire other dimension is stronger than needed.
                return self(args[0], axis, start, end)
            if not isinstance(old[axis], int) or any(x is not None and not isinstance(x, int) for x in (lo, hi)):
                return UNKNOWN
            offset, _, _ = slice(lo, hi).indices(old[axis])
            return self(args[0], axis, start + offset, end + offset)
        if target == 'aten.cat.default':
            dim = (args[1] if len(args) > 1 else 0) % len(dims)
            values, offset = [], 0
            for source in args[0]:
                if dim != axis:
                    values.append(self(source, axis, start, end))
                    continue
                width = shape(source)[axis]
                if not isinstance(width, int):
                    return UNKNOWN
                lo, hi = max(start, offset), min(end, offset + width)
                if lo < hi:
                    values.append(self(source, axis, lo-offset, hi-offset))
                offset += width
            return self.equal_values(values)
        if target == 'aten.stack.default':
            dim = (args[1] if len(args) > 1 else 0) % len(dims)
            if dim == axis:
                # A selected feature must be uniform over all its coordinates.
                return UNKNOWN
            return self.equal_values([self(source, axis-int(dim < axis), start, end)
                                      for source in args[0]])
        if n.target is operator.getitem and isinstance(args[0], Node) and str(args[0].target) == 'aten.unbind.int':
            unbind, index = args
            source = unbind.args[0]
            old = shape(source)
            if old is None or not isinstance(index, int):
                return UNKNOWN
            dim = (unbind.args[1] if len(unbind.args) > 1 else 0) % len(old)
            if str(source.target) == 'aten.stack.default':
                stacked_dim = (source.args[1] if len(source.args) > 1 else 0) % len(old)
                if stacked_dim == dim:
                    return self(source.args[0][index], axis, start, end)
            # For a constructor, uniformity across ALL unbound rows suffices.
            return self(source, axis + int(dim <= axis), start, end)
        boolean_binary = {'aten.logical_and.default', 'aten.logical_or.default',
                          'aten.__and__.Tensor', 'aten.__or__.Tensor',
                          'aten.bitwise_and.Tensor', 'aten.bitwise_or.Tensor'}
        if target in boolean_binary:
            if n.meta['val'].dtype != torch.bool:
                return UNKNOWN
            left = self.operand(args[0], n, axis, start, end)
            right = self.operand(args[1], n, axis, start, end)
            is_and = 'and' in target
            absorbing = False if is_and else True
            # Boolean absorption permits unknown feature fields safely.
            if any(v is not UNKNOWN and bool(v) == absorbing for v in (left, right)):
                return absorbing
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            return bool(left) and bool(right) if is_and else bool(left) or bool(right)
        if target in {'aten.bitwise_not.default', 'aten.logical_not.default'}:
            if n.meta['val'].dtype != torch.bool:
                return UNKNOWN
            value = self.operand(args[0], n, axis, start, end)
            return not bool(value) if value is not UNKNOWN else UNKNOWN
        if target in {'aten.eq.Tensor', 'aten.eq.Scalar', 'aten.ne.Tensor', 'aten.ne.Scalar',
                      'aten.add.Tensor', 'aten.add.Scalar'}:
            left = self.operand(args[0], n, axis, start, end)
            right = self.operand(args[1], n, axis, start, end)
            if left is UNKNOWN or right is UNKNOWN:
                return UNKNOWN
            # Use actual ATen promotion/rounding, rather than Python arithmetic.
            # A common exported sentinel is zeros(shape) + (-1). Both operands
            # must be uniformly known; no arithmetic absorption is assumed.
            def tensor(value, original):
                if not isinstance(original, Node):
                    return value
                meta = original.meta['val']
                return torch.full((1,) if meta.ndim else (), value, dtype=meta.dtype, device='cpu')
            if any(isinstance(value, Node) for value in args[2:]):
                return UNKNOWN
            if any(isinstance(value, Node) for value in n.kwargs.values()):
                return UNKNOWN
            return n.target(tensor(left, args[0]), tensor(right, args[1]),
                            *args[2:], **n.kwargs).item()
        return UNKNOWN

    @staticmethod
    def equal_values(values):
        if not values or any(v is UNKNOWN for v in values):
            return UNKNOWN
        return values[0] if all(v == values[0] for v in values) else UNKNOWN
