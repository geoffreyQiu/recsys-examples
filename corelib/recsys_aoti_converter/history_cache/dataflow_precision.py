# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Graph-visible BF16 policy, distinct from kernel-internal accumulation.

Capture the unchanged source in FP32 first: source-level .float() calls can
otherwise make BF16 parameters incompatible with their inputs. Lower the
captured ATen graph, not the original Python source, then export it again
to recompute all tensor metadata and validate operator dtype compatibility.
"""
from collections import Counter
import copy
import operator

import torch
from torch.utils import _pytree as pytree

from .precision import precision_name


def validate_autocast(precision, autocast=False):
    if not isinstance(autocast, bool):
        raise ValueError('autocast must be a boolean')
    if autocast and precision_name(precision) != 'bf16':
        raise ValueError('--autocast is an option for --precision bf16 only')
    return autocast


def bf16_inputs(inputs):
    """Lower ordinary floats; preserve explicit FP64 and integer/bool inputs."""
    return pytree.tree_map(
        lambda x: x.to(dtype=torch.bfloat16) if isinstance(x, torch.Tensor)
        and x.is_floating_point() and x.dtype != torch.float64 else x, inputs)


def lower_bf16(exported):
    """Return a BF16 GraphModule and examples without changing source weights.

    Explicit FP16/FP32 dtype requests become BF16. Explicit FP64 requests and
    their downstream type promotion are preserved, including division helpers.
    Operators may still accumulate internally in FP32.
    """
    module = copy.deepcopy(exported.module())
    module._apply(lambda x: x.to(dtype=torch.bfloat16)
                  if x.is_floating_point() and x.dtype != torch.float64 else x)
    for node in list(module.graph.nodes):
        if node.op != 'call_function':
            continue
        def dtype(value):
            return torch.bfloat16 if value in (torch.float16, torch.float32) else value
        node.args = pytree.tree_map(dtype, node.args)
        node.kwargs = pytree.tree_map(dtype, node.kwargs)
        # Integer features combined with a Python floating scalar otherwise
        # silently create FP32 tensors even with autocast disabled. Express
        # the requested arithmetic policy at the first floating operation.
        result = node.meta.get('val')
        factories = {'aten.zeros.default', 'aten.ones.default', 'aten.empty.memory_format',
            'aten.full.default', 'aten.scalar_tensor.default', 'aten.arange.default',
            'aten.arange.start', 'aten.arange.start_step', 'aten.eye.default'}
        if (str(node.target) in factories and isinstance(result, torch.Tensor)
                and result.is_floating_point()
                and node.kwargs.get('dtype') != torch.float64):
            node.kwargs = {**node.kwargs, 'dtype': torch.bfloat16}
        if str(node.target) == 'aten.where.Scalar' and isinstance(result, torch.Tensor) and result.is_floating_point():
            condition, left, right = node.args
            with module.graph.inserting_before(node):
                left = module.graph.call_function(torch.ops.aten.full_like.default,
                    (condition, left), {'dtype': torch.bfloat16})
                right = module.graph.call_function(torch.ops.aten.full_like.default,
                    (condition, right), {'dtype': torch.bfloat16})
            node.target = torch.ops.aten.where.self
            node.args = (condition, left, right)
        inputs = [x for x in node.all_input_nodes
                  if isinstance(x.meta.get('val'), torch.Tensor)]
        implicit_float = {'aten.add.Tensor', 'aten.sub.Tensor', 'aten.mul.Tensor',
            'aten.div.Tensor', 'aten.div.Scalar', 'aten.log.default',
            'aten.log1p.default', 'aten.sqrt.default', 'aten.pow.Tensor_Scalar'}
        if (str(node.target) in implicit_float and isinstance(result, torch.Tensor)
                and result.is_floating_point() and inputs
                and all(not x.meta['val'].is_floating_point() for x in inputs)):
            source = inputs[0]
            with module.graph.inserting_before(node):
                cast = module.graph.call_function(torch.ops.aten._to_copy.default,
                    (source,), {'dtype': torch.bfloat16})
            node.replace_input_with(source, cast)
    module.graph.lint()
    module.recompile()
    return module, bf16_inputs(exported.example_inputs[0])


def audit_dataflow(exported, *, require_bf16=False):
    """Inspect actual ExportedProgram metadata, parameters, constants and ABI.

    Do not infer dtypes from source-code strings. Tuple-valued ops (including
    custom cache reads) and graph outputs are inspected recursively too.
    """
    counts, float_nodes, casts, empty_values, unused_values = Counter(), [], [], [], []
    for node in exported.graph.nodes:
        val = node.meta.get('val')
        leaves = pytree.tree_leaves(val)
        tuple_getters = isinstance(val, (list, tuple)) and all(
            u.target is operator.getitem and isinstance(u.args[1], int) for u in node.users)
        used_indices = {u.args[1] for u in node.users if u.users} if tuple_getters else None
        for index, tensor in enumerate(leaves):
            if not isinstance(tensor, torch.Tensor): continue
            counts[str(tensor.dtype)] += 1
            if tensor.is_floating_point():
                if ((node.op != 'output' and not node.users)
                        or used_indices is not None and index not in used_indices):
                    unused_values.append({'node': node.name, 'index': index, 'dtype': str(tensor.dtype), 'op': str(node.target)})
                    continue
                if any(isinstance(d, int) and d == 0 for d in tensor.shape):
                    empty_values.append({'node': node.name, 'dtype': str(tensor.dtype), 'op': str(node.target)})
                    continue
                float_nodes.append({'node': node.name, 'op': str(node.target),
                                    'dtype': str(tensor.dtype)})
        if node.op == 'call_function' and str(node.target) in (
                'aten._to_copy.default', 'aten.to.dtype', 'aten.to.dtype_layout',
                'prims.convert_element_type.default') and node.args:
            source = getattr(node.args[0], 'meta', {}).get('val')
            dest = node.meta.get('val')
            if (isinstance(source, torch.Tensor) and isinstance(dest, torch.Tensor)
                    and source.is_floating_point() and dest.is_floating_point()
                    and source.dtype != dest.dtype):
                casts.append({'node': node.name, 'from': str(source.dtype),
                              'to': str(dest.dtype)})
    states = {name: str(value.dtype) for name, value in exported.state_dict.items()
              if isinstance(value, torch.Tensor) and value.is_floating_point()}
    constants = {name: str(value.dtype) for name, value in exported.constants.items()
                 if isinstance(value, torch.Tensor) and value.is_floating_point()}
    inputs = [str(x.dtype) for x in pytree.tree_leaves(exported.example_inputs)
              if isinstance(x, torch.Tensor)]
    non_bf16 = [x for x in float_nodes if x['dtype'] != 'torch.bfloat16']
    bf16_ok = (not non_bf16 and not casts
               and all(d == 'torch.bfloat16' for d in (*states.values(), *constants.values())))
    allowed = {'torch.bfloat16', 'torch.float64'}
    unexpected = [x for x in float_nodes if x['dtype'] not in allowed]
    unexpected_casts = [x for x in casts if x['from'] not in allowed or x['to'] not in allowed]
    policy_ok = (not unexpected and not unexpected_casts
                 and all(d in allowed for d in (*states.values(), *constants.values())))
    report = {'node_tensor_dtypes': dict(counts), 'floating_node_values': len(float_nodes),
              'floating_state_dtypes': dict(Counter(states.values())),
              'floating_constant_dtypes': dict(Counter(constants.values())),
              'input_dtypes': inputs, 'floating_dtype_transitions': casts,
              'non_bf16_floating_nodes': non_bf16,
              'empty_auxiliary_values': empty_values,
              'unconsumed_auxiliary_values': unused_values,
              'bf16_graph_dataflow_passed': bf16_ok,
              'bf16_policy_dataflow_passed': policy_ok,
              'precision_policy': 'BF16 except preserved explicit FP64 requests and their promoted dataflow',
              'scope': 'graph-visible tensors; kernel-internal accumulation is unrestricted'}
    if require_bf16 and not policy_ok:
        raise ValueError(f'BF16/explicit-FP64 dataflow audit failed: {unexpected[:8]}, casts={unexpected_casts[:8]}')
    return report
