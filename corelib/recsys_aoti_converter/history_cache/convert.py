# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Structural K/V matching over exported ATen graphs; no model-class matching."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import torch
from torch.fx import Graph, GraphModule, Node
from .slicing import ConversionError, TokenSlicer, dependencies, eliminate_common_subexpressions


@dataclass(frozen=True)
class CacheSpec:
    history_input_indices: tuple[int, ...]
    length: int
    heads: int
    head_dim: int
    policy: str = "immutable_snapshot"
    history_start: int = 0


@dataclass
class Partition:
    producer: GraphModule
    consumer: GraphModule
    producer_input_indices: list[int]
    report: dict

    @property
    def banks(self):
        return len(self.report["banks"])


def _ancestors(nodes):
    seen = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(n.all_input_nodes)
    return seen


def _shape(n):
    value = n.meta.get("val")
    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _linear_source(n):
    if n.op != "call_function" or n.target != torch.ops.aten.permute.default or list(n.args[1]) != [0, 2, 1, 3]:
        return None
    reshape = n.args[0]
    if reshape.op != "call_function" or reshape.target not in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
        return None
    linear = reshape.args[0]
    if linear.op == "call_function" and linear.target == torch.ops.aten.linear.default:
        return linear
    return None


def _projection_input_key(node):
    # Autocast lowering may emit separate identical casts for K and V. Match
    # their complete cast chains and shared source, without dropping casts or
    # treating different precision/rounding policies as interchangeable.
    if node.op == 'call_function' and node.target in (
            torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype):
        return (node.target, _projection_input_key(node.args[0]),
                node.args[1:], tuple(sorted(node.kwargs.items())))
    return node


def _attention_role(boundary):
    """Recognize K/V use through retained layout/cast ops on a matmul RHS."""
    transparent = {
        torch.ops.aten.unsqueeze.default, torch.ops.aten.expand.default,
        torch.ops.aten.reshape.default, torch.ops.aten.view.default,
        torch.ops.aten.clone.default, torch.ops.aten.contiguous.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype,
    }
    seen, roles = set(), set()
    stack = [(boundary, False)]
    while stack:
        current, transposed = stack.pop()
        if (current, transposed) in seen:
            continue
        seen.add((current, transposed))
        for user in current.users:
            if user.op != "call_function":
                continue
            if user.target == torch.ops.aten.transpose.int and set(user.args[1:]) == {-1, -2}:
                stack.append((user, not transposed))
            elif user.target in transparent:
                stack.append((user, transposed))
            elif user.target in (torch.ops.aten.matmul.default, torch.ops.aten.bmm.default) and user.args[1] is current:
                roles.add("key" if transposed else "value")
            else:
                raise ConversionError(f"unsupported_memory_consumer: {current.name} -> {user.target}")
    return roles


def partition_history(exported: torch.export.ExportedProgram, spec: CacheSpec) -> Partition:
    from .layered import analyze_layered
    layered = analyze_layered(exported, spec)
    if layered is not None:
        return layered
    if spec.policy != "immutable_snapshot":
        raise ConversionError("unsupported_cache_policy: only immutable_snapshot is supported")
    module = exported.module()
    nodes = list(module.graph.nodes)
    inputs = [n for n in nodes if n.op == "placeholder"]
    if (not spec.history_input_indices or min(spec.history_input_indices) < 0
            or max(spec.history_input_indices) >= len(inputs)
            or len(set(spec.history_input_indices)) != len(spec.history_input_indices)
            or min(spec.length, spec.heads, spec.head_dim) <= 0):
        raise ConversionError("invalid_history_inputs")
    history = {inputs[i] for i in spec.history_input_indices}
    if any(_shape(n) is None or len(_shape(n)) < 2 or _shape(n)[1] != spec.length for n in history):
        raise ConversionError("invalid_history_inputs: declared history must have token axis 1 and the fixed length")
    deps = dependencies(nodes)
    candidates = []
    for n in nodes:
        dims, linear = _shape(n), _linear_source(n)
        if (linear is not None and dims and len(dims) == 4
                and dims[1:] == (spec.heads, spec.length, spec.head_dim)
                and deps[n] & history):
            candidates.append((n, linear, _attention_role(n)))
    pairs = [(k, v) for k, kl, kr in candidates for v, vl, vr in candidates
             if k is not v and kr == {"key"} and vr == {"value"}
             and _projection_input_key(kl.args[0]) == _projection_input_key(vl.args[0])]
    if not pairs or len({n for pair in pairs for n in pair}) != 2 * len(pairs):
        raise ConversionError(f"ambiguous_or_missing_kv_pair: found {len(pairs)} unambiguous shared-input pairs")

    # Prove prefix purity at each token, not merely for the original step's
    # full tensor. Position-affine weights are part of each sliced expression.
    analysis = TokenSlicer(deps, history)
    banks, producer_dependencies = [], set()
    for key, value in pairs:
        prefix, offsets = 0, set()
        for position in range(spec.length):
            k = analysis.select(key, 2, position, position + 1)
            v = analysis.select(value, 2, position, position + 1)
            required = k.deps | v.deps
            if not required or not required <= history:
                break
            if k.origins != v.origins:
                raise ConversionError("inconsistent_kv_token_sources: " + key.name)
            offsets.update(source_position - position for _, source_position in k.origins)
            producer_dependencies.update(required)
            prefix += 1
        if prefix == 0:
            raise ConversionError("no_history_only_prefix: " + key.name)
        # Validate the request tail with the same token-local interpreter.
        # Unknown token-mixing operations cannot silently become cacheable.
        if prefix < spec.length:
            analysis.select(key, 2, prefix, spec.length)
            analysis.select(value, 2, prefix, spec.length)
        banks.append({"key_node": key.name, "value_node": value.name,
                      "prefix_length": prefix, "request_tail_length": spec.length - prefix,
                      "history_source_offsets": sorted(offsets),
                      "position_range": [0, prefix], "storage_length": spec.length})

    producer_inputs = [i for i, n in enumerate(inputs) if n in producer_dependencies]
    graph, env = Graph(), {}
    for i in producer_inputs:
        env[inputs[i]] = graph.placeholder(inputs[i].name)
    producer_slicer = TokenSlicer(deps, history, graph, env)
    keys, values = [], []
    for (key, value), bank in zip(pairs, banks):
        prefix = bank["prefix_length"]
        for source, outputs in ((key, keys), (value, values)):
            selected = producer_slicer.select(source, 2, 0, prefix)
            if not selected.deps <= history:
                raise ConversionError("undeclared_request_dependency in generated producer")
            result = selected.node
            if prefix < spec.length:
                result = graph.call_function(torch.ops.aten.constant_pad_nd.default,
                    (result, [0, 0, 0, spec.length - prefix], 0.0))
            outputs.append(result)
    # Packed along L for the API; native storage uses one fixed-L bank per pair.
    packed = [parts[0] if len(parts) == 1 else graph.call_function(torch.ops.aten.cat.default, (parts, 2))
              for parts in (keys, values)]
    graph.output(tuple(packed))
    producer_cse = eliminate_common_subexpressions(graph)
    producer = GraphModule(module, graph, "HistoryProducer")

    graph, env = Graph(), {}
    for n in inputs:
        env[n] = graph.placeholder(n.name)
        env[n].meta = dict(n.meta)
    cached_key = graph.placeholder("cached_history_key")
    cached_value = graph.placeholder("cached_history_value")
    consumer_slicer = TokenSlicer(deps, history, graph, env)
    for index, ((key, value), bank) in enumerate(zip(pairs, banks)):
        prefix = bank["prefix_length"]
        for source, packed_input in ((key, cached_key), (value, cached_value)):
            begin = index * spec.length
            shared_prefix = graph.call_function(torch.ops.aten.slice.Tensor,
                (packed_input, 2, begin, begin + prefix))
            if prefix < spec.length:
                tail = consumer_slicer.select(source, 2, prefix, spec.length).node
                # New request-local tensor: never write feedback to shared slots.
                env[source] = graph.call_function(torch.ops.aten.cat.default, ([shared_prefix, tail], 2))
            else:
                env[source] = shared_prefix
            bank['consumer_' + ('key_node' if source is key else 'value_node')] = env[source].name
    for n in nodes:
        if n not in env:
            env[n] = graph.node_copy(n, lambda x: env[x])
    consumer_cse = eliminate_common_subexpressions(graph)
    consumer = GraphModule(module, graph, "HistoryConsumer")
    graph.eliminate_dead_code()
    consumer.recompile()
    producer.graph.lint()
    consumer.graph.lint()
    report = {"spec": asdict(spec), "banks": banks, "bank_count": len(banks),
              "key_node": pairs[0][0].name, "value_node": pairs[0][1].name,
              "producer_inputs": producer_inputs,
              "original_nodes": len(nodes), "producer_nodes": len(list(producer.graph.nodes)),
              "consumer_nodes": len(list(consumer.graph.nodes)),
              "producer_cse_removed": producer_cse, "consumer_cse_removed": consumer_cse,
              "pattern": "shared-input linear K/V; static token-local dependency slicing; history-only prefix + request-local tail",
              "cached_prefix_rows": sum(b["prefix_length"] for b in banks),
              "storage_rows_per_history": len(banks) * spec.length,
              "consumer_memory_length": spec.length,
              "feedback_storage": "request-local concatenation; shared GPU slots are read-only during consumption",
              "execution": "common request dispatcher + AOTI producer/consumer; structural matching only"}
    return Partition(producer, consumer, producer_inputs, report)
