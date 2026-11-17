# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Replay fixed batch/missing-row shapes with real cache control per request.

Every captured body comes from the same compiled package. Snapshot lookup,
context validation, allocation, transfer completion, offload and retirement are
never replayed from an old request. Static capture slots are updated every time.
"""
from pathlib import Path
import json
import os

import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils import _pytree as pt


def extract_context(bundle, manifest):
    """Extract only the pure input-dependent argument to history prepare."""
    if not manifest.get('context_guard'):
        return None, []
    original = torch.export.load(Path(bundle) / 'model_export.pt2').module()
    inputs = [n for n in original.graph.nodes if n.op == 'placeholder']
    prepares = [n for n in original.graph.nodes if n.op == 'call_function'
                and n.target == torch.ops.history_single.prepare.default]
    if len(prepares) != 1 or len(prepares[0].args) != 5:
        raise ValueError('A unique explicit context guard is required for CUDA replay')
    context = prepares[0].args[4]
    graph, env = Graph(), {}
    first = 2 + manifest['bank_count']
    indices = []
    needed = set()
    def visit(n):
        if not isinstance(n, Node) or n in needed:
            return
        needed.add(n)
        for x in n.all_input_nodes:
            visit(x)
    visit(context)
    for index, n in enumerate(inputs):
        if n not in needed:
            continue
        if index < first:
            raise ValueError('Context must depend only on original request inputs')
        indices.append(index - first)
        env[n] = graph.placeholder(n.name)
    for n in original.graph.nodes:
        if n not in needed or n in env:
            continue
        if n.op == 'call_function' and ('history_single' in str(n.target) or 'with_effects' in str(n.target)):
            raise ValueError('Side effects cannot enter context extraction')
        if n.op not in ('get_attr', 'call_function'):
            raise ValueError(f'Unsupported context dependency: {n.op}')
        env[n] = graph.node_copy(n, lambda x: env[x])
    graph.output(env[context])
    graph.lint()
    return GraphModule(original, graph, 'CacheContext'), indices


class CachedCudaGraphs:
    def __init__(self, runner, *, context_aoti=False, overlap_onboard=None):
        self.runner = runner
        self.cache = runner.cache
        self.manifest = runner.manifest
        self.context, self.context_indices = extract_context(runner.bundle, self.manifest)
        if context_aoti and self.context is not None:
            path = runner.bundle / 'context_guard'
            metadata = json.loads((path / 'manifest.json').read_text())
            if (metadata['model_namespace'] != self.manifest['namespace'] or
                    metadata['torch'] != torch.__version__ or
                    metadata['batch_range'] != self.manifest['batch_range'] or
                    tuple(metadata['gpu_arch']) != torch.cuda.get_device_capability() or
                    metadata['indices'] != self.context_indices or not metadata['passed']):
                raise ValueError('Compiled cache-context check does not match this model/runtime')
            self.context = torch._inductor.aoti_load_package(str(path / 'context_aoti.pt2'),
                                                            run_single_threaded=True)
        self.graphs = {}
        self.captures = self.replays = self.fallbacks = 0
        if overlap_onboard is None:
            overlap_onboard = os.environ.get('HISTORY_ONBOARD_OVERLAP', '1') != '0'
        op = torch.ops.history_single
        self.overlap_onboard = bool(overlap_onboard and getattr(self.cache, 'owner_loop', False)
                                    and hasattr(op, 'prepare_replay_async')
                                    and hasattr(op, 'split_graph'))
        self.split_captures = self.overlap_replays = self.full_replays = 0
        self.split_shapes = {}

    @property
    def last_stats(self):
        return self.runner.last_stats

    def _capture(self, ticket, rows, slots, missing, flat, ids):
        op = torch.ops.history_single
        op.capture_barrier(ticket)
        static_slots = slots.clone()
        static_rows = rows[:missing].to(device=slots.device)
        inputs = list(flat)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op.capture_begin(ticket, static_slots, static_rows)
                try:
                    self.runner.model(ids, torch.tensor([False]), *self.cache.tables, *inputs)
                finally:
                    op.capture_end(ticket)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=self.overlap_onboard)
        op.capture_begin(ticket, static_slots, static_rows)
        try:
            if self.overlap_onboard:
                op.capture_overlap_mark(True)
            with torch.cuda.graph(graph):
                outputs = self.runner.model(ids, torch.tensor([False]), *self.cache.tables, *inputs)
        finally:
            if self.overlap_onboard:
                op.capture_overlap_mark(False)
            op.capture_end(ticket)
        split_handle = None
        if self.overlap_onboard:
            split = op.split_graph(graph.raw_cuda_graph()).tolist()
            split_handle = split[0]
            self.split_shapes[(len(ids), missing)] = split[1:]
            self.split_captures += 1
            graph.instantiate()
        self.captures += 1
        return graph, inputs, static_slots, static_rows, outputs, split_handle

    def close(self):
        """Release fragment executables before their captured tensor storage."""
        op = torch.ops.history_single
        for record in self.graphs.values():
            if record[5] is not None:
                op.destroy_split(record[5])
        self.graphs.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Native shutdown also cleans the thread-local executable registry.
            pass

    @torch.inference_mode()
    def __call__(self, *args, snapshot_ids, offload=False, **kwargs):
        flat, ids = self.runner.validate_inputs(args, kwargs, snapshot_ids)
        return pt.tree_unflatten(self._run_flat(flat, ids, offload), self.runner.output_tree)

    def validate_flat_inputs(self, flat, snapshot_ids, *, check_tensor_contract=True):
        return self.runner.validate_flat_inputs(flat, snapshot_ids, check_tensor_contract=check_tensor_contract)

    @torch.inference_mode()
    def run_flat(self, flat, *, snapshot_ids, offload=False, check_tensor_contract=True):
        flat, ids = self.validate_flat_inputs(flat, snapshot_ids, check_tensor_contract=check_tensor_contract)
        return self._run_flat(flat, ids, offload)

    def _run_flat(self, flat, ids, offload):
        base, op = self.runner, torch.ops.history_single
        batch = flat[0].shape[0]
        with self.cache.lock:
            base.verify_snapshots(flat, ids)
            try:
                context = self.context(*[flat[i] for i in self.context_indices]) if self.context is not None else None
                if self.overlap_onboard or (getattr(self.cache, 'resident_fastpath', False) and hasattr(op, 'prepare_replay')):
                    existing = self.graphs.get((batch, 0))
                    destination = existing[2] if existing is not None else None
                    prepare = op.prepare_replay_async if self.overlap_onboard else op.prepare_replay
                    ticket, rows, slots, info = prepare(
                        ids, flat[0], self.cache.length, self.cache.banks, context, destination)
                    info = info.tolist()
                else:
                    ticket, rows, slots = op.prepare(ids, flat[0], self.cache.length, self.cache.banks, context)
                    info = op.prepared_info(ticket).tolist()
                    op.ready_for_replay(ticket)
                shape = (batch, info[1])
                if shape not in self.graphs:
                    # Capture itself is initialization, never part of steady-state
                    # overlap. All real transfers must finish before capture.
                    if self.overlap_onboard:
                        op.ready_for_replay(ticket)
                    self.graphs[shape] = self._capture(ticket, rows, slots, info[1], flat, ids)
                graph, inputs, static_slots, static_rows, result, split_handle = self.graphs[shape]
                for dst, src in zip(inputs, flat):
                    if dst.data_ptr() != src.data_ptr():
                        dst.copy_(src, non_blocking=True)
                if static_slots.data_ptr() != slots.data_ptr():
                    static_slots.copy_(slots, non_blocking=True)
                if info[1]:
                    # The producer receives precisely the current missing rows,
                    # including different permutations with the same miss count.
                    # The source CPU tensor stays alive until retirement below.
                    static_rows.copy_(rows[:info[1]], non_blocking=True)
                if self.overlap_onboard and info[2]:
                    # The capture marker is immediately before the first cache
                    # read. The prefix has no dependence on onboarded K/V.
                    op.replay_prefix(split_handle)
                    op.ready_for_replay(ticket)
                    op.replay_suffix(split_handle, ticket)
                    self.overlap_replays += 1
                else:
                    if self.overlap_onboard:
                        op.ready_for_replay(ticket)
                    graph.replay()
                    self.full_replays += 1
                final = op.replay_finish(ticket, result[-2], torch.tensor([offload]))
                base.record_stats(final)
                self.replays += 1
                return list(result[:-1])
            except BaseException:
                base.failed = self.cache.failed = True
                raise
