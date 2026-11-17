# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free, real-FlexKV paged owner lifecycle and AOTI smoke test.

Run explicitly, outside CUDA graph capture. This starts its own FlexKV server,
uses one CUDA device, and writes a JSON result plus the optional tiny PT2.
No HSTU checkpoint, embeddings, Triton server, or converter is required.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

PAGE = 32
LAYERS = 2
HEADS = 2
DIM = 16
PAGES = 12


def start_server(directory: Path, owner_loop: bool):
    os.environ["FLEXKV_ENABLE_MPS"] = "0"
    os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "1"
    from flexkv.common import config as flexkv_config
    from flexkv.common.config import CacheConfig, ModelConfig
    from flexkv.server.server import KVServer

    model = ModelConfig(num_layers=LAYERS, num_kv_heads=HEADS, head_size=DIM,
                        dtype=torch.bfloat16, tp_size=1, dp_size=1)
    # Explicit small capacities work with both adapter revisions; this fixture
    # does not require the newer optional RankInfo/UserConfig convenience API.
    cache = CacheConfig(tokens_per_block=PAGE, num_cpu_blocks=1024,
                        num_tmp_cpu_blocks=16, num_local_blocks=1024,
                        enable_ssd=False, enable_p2p_cpu=False, enable_p2p_ssd=False)
    receive = f"ipc://{directory}/server.sock"
    register = f"ipc://{directory}/register.sock"
    settings = {
        "server_recv_port": receive, "gpu_register_port": register,
        "num_layers": LAYERS, "num_kv_heads": HEADS, "head_size": DIM,
        "tokens_per_page": PAGE, "tokens_per_chunk": PAGE,
        "num_primary_cache_pages": PAGES, "num_buffer_pages": 0,
        "max_batch_size": 4, "max_sequence_length": 128, "device_idx": 0,
        "dtype": "bfloat16", "host_capacity_per_layer": 32 * 1024 * 1024,
        "host_kvstorage_backend": "flexkv", "owner_loop": int(owner_loop),
        "onboard_child_tasks": int(not hasattr(flexkv_config, "RankInfo")),
        "fixed_history_slots": 0, "onload_timeout_ms": 10000.0,
        "offload_timeout_ms": 10000.0, "offload_mode": "lazy",
        "host_kvstorage_fail_policy": "fail_close",
    }
    path = directory / "runtime.yaml"
    path.write_text("".join(f"{key}: {value}\n" for key, value in settings.items()))
    os.environ["KVCACHE_MANAGER_CONFIG_FILE"] = str(path)
    server = KVServer.create_server(
        model_config=model, cache_config=cache, server_recv_port=receive,
        gpu_register_port=register, total_clients=1, inherit_env=True,
    )
    # The helper used by the HSTU demo uses the same startup allowance.
    time.sleep(3)
    return server


class PagedFixture:
    def __init__(self, *, owner_loop=True):
        self.owner_loop = owner_loop
        self.ops = torch.ops.kvcache_manager_ops
        self.anchor = torch.empty(0, dtype=torch.int64)
        self.ops.init_kvcache(self.anchor)

    def request(self, ids, lengths):
        ids = torch.tensor(ids, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        lookup = self.ops.lookup(ids, lengths, ids)
        allocation = self.ops.allocate(ids, lengths, lookup[1], lookup[5])
        launched = self.ops.onboard_launch(ids, lengths, lookup, allocation[0], allocation[2])
        tables = self.ops.onboard_wait(launched[2], allocation[0])
        return ids, lengths, lookup, allocation, tables

    @staticmethod
    def coordinates(request, row):
        allocation = request[3]
        offsets = allocation[2].cpu().tolist()
        pages = allocation[0][offsets[row]:offsets[row + 1]].long()
        positions = torch.arange(int(request[1][row]), device="cuda")
        return pages[positions // PAGE], positions % PAGE

    @staticmethod
    def expected(uid, length, layer):
        # Exact BF16 values, distinct by user, layer, token, K/V and channel.
        token = torch.arange(length, device="cuda").remainder(7).view(-1, 1, 1, 1)
        kind = torch.arange(2, device="cuda").view(1, 2, 1, 1)
        channel = torch.arange(DIM, device="cuda").remainder(4).view(1, 1, 1, -1)
        return ((uid % 8) * 16 + layer * 4 + token + kind + channel / 4).expand(
            length, 2, HEADS, DIM
        ).to(torch.bfloat16)

    def fill_missing(self, request, *, delay=False):
        for row, (uid, length) in enumerate(zip(request[0].tolist(), request[1].tolist())):
            cached = int(request[2][1][row])
            page, position = self.coordinates(request, row)
            # Prepare allocations before delaying the actual writes. No explicit
            # sync may occur between those writes and offload_launch in the test.
            expected = [self.expected(uid, length, layer) for layer in range(LAYERS)]
            if delay:
                torch.cuda._sleep(20_000_000)
            for layer, table in enumerate(request[4]):
                table[page[cached:], :, position[cached:], :, :] = expected[layer][cached:]
        return self.ops.request_finish(request[4][0][0, 0, 0, 0, :])

    def check(self, request):
        for row, (uid, length) in enumerate(zip(request[0].tolist(), request[1].tolist())):
            page, position = self.coordinates(request, row)
            for layer, table in enumerate(request[4]):
                actual = table[page, :, position, :, :]
                torch.testing.assert_close(actual, self.expected(uid, length, layer), rtol=0, atol=0)
        return self.ops.request_finish(request[4][0][0, 0, 0, 0, :])

    def offload(self, request, dependency, *, explicit_mappings=False):
        ids, lengths, lookup, allocation, _ = request
        mappings = [torch.tensor([-1], dtype=torch.int64)]
        if explicit_mappings:
            pages, offsets = allocation[0].cpu().long(), allocation[2].cpu().tolist()
            mappings = [
                (pages[offsets[row]:offsets[row + 1], None] * PAGE
                 + torch.arange(PAGE)[None, :]).flatten()[:length]
                for row, length in enumerate(lengths.tolist())
            ]
        return self.ops.offload_launch(
            ids, lengths, lookup[1], lookup[5], lookup[2], lookup[3],
            allocation[0], allocation[2], mappings, dependency,
        )

    def evict(self, ids):
        self.ops.evict_kvcache(torch.tensor(ids, dtype=torch.int64), True, self.anchor)

    def flush(self, token):
        self.ops.offload_flush(token)


class ReadPaged(torch.nn.Module):
    """One complete paged request in native AOTI, with live completion op."""

    def forward(self, ids, lengths, independent):
        ops = torch.ops.kvcache_manager_ops
        lookup = ops.lookup(ids, lengths, ids)
        allocation = ops.allocate(ids, lengths, lookup[1], lookup[5])
        launched = ops.onboard_launch(ids, lengths, lookup, allocation[0], allocation[2])
        independent = independent.sin()
        tables = ops.onboard_wait(launched[2], independent)
        # This smoke artifact handles one 64-token request (two actual pages).
        torch._check(allocation[0].numel() == 2)
        pages = allocation[0].long()
        result = tables[0].index_select(0, pages).float().sum().reshape(1)
        result = result + independent.sum()
        return ops.request_finish(result)


class PublishPaged(torch.nn.Module):
    """Like HSTU, retain offload through the completion op's dependency edge."""

    def forward(self, ids, lengths, merged, host, start, cached, pages, offsets, dependency):
        ticket = torch.ops.kvcache_manager_ops.offload_launch(
            ids, lengths, merged, host, start, cached, pages, offsets,
            [torch.tensor([-1], dtype=torch.int64)], dependency,
        )
        return torch.ops.kvcache_manager_ops.request_finish(dependency, ticket)


@torch.inference_mode()
def aoti_smoke(fixture, output):
    from recsys_kvcache_manager.fake_kvcache_manager_ops import register_fake_kvcache_manager_ops
    assert register_fake_kvcache_manager_ops()
    seeded = fixture.request([101], [64])
    dependency = fixture.fill_missing(seeded)
    ids, lengths, lookup, allocation, _ = seeded
    publish_inputs = (ids, lengths, lookup[1], lookup[5], lookup[2], lookup[3],
                      allocation[0], allocation[2], dependency)
    publication = torch.export.export(PublishPaged(), publish_inputs, strict=False)
    assert any("offload_launch" in str(node.target) for node in publication.graph.nodes)
    assert any("request_finish" in str(node.target) for node in publication.graph.nodes)
    torch.export.save(publication, output / "publish.exported.pt2")
    publish_path = torch._inductor.aoti_compile_and_package(
        publication, package_path=str(output / "publish.pt2")
    )
    publisher = torch._inductor.aoti_load_package(publish_path)
    fixture.flush(publisher(*publish_inputs))
    read_inputs = (ids, lengths, torch.ones(8, device="cuda"))
    exported = torch.export.export(ReadPaged(), read_inputs, strict=False)
    targets = [str(node.target) for node in exported.graph.nodes]
    for name in ("lookup", "allocate", "onboard_launch", "onboard_wait", "request_finish"):
        assert any(f"kvcache_manager_ops.{name}." in target for target in targets), name
    (output / "read_graph.py").write_text(exported.graph_module.code)
    package = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(output / "read.pt2")
    )
    compiled = torch._inductor.aoti_load_package(package)
    expected = fixture.expected(101, 64, 0).float().sum() + read_inputs[2].sin().sum()
    gpu = compiled(*read_inputs)
    fixture.evict([101])
    cpu = compiled(*read_inputs)
    torch.testing.assert_close(gpu.flatten(), expected.reshape(1), rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(cpu, gpu, rtol=0, atol=0)
    assert gpu.data_ptr() != cpu.data_ptr(), "request_finish must return fresh output storage"


@torch.inference_mode()
def lifecycle(fixture):
    checks = []
    request = fixture.request([11, 22], [64, 64])
    assert request[2][1].tolist() == [0, 0]
    dependency = fixture.fill_missing(request)
    fixture.check(request)
    checks.append("cold_fill")

    hit = fixture.request([11, 22], [64, 64])
    assert hit[2][1].tolist() == [64, 64]
    fixture.check(hit)
    checks.append("gpu_hit")

    # Two CPU hits exercise batched/merged FlexKV task completion, including
    # child-record cleanup in the pinned adapter (a single CPU hit cannot).
    dependency = fixture.check(hit)
    fixture.flush(fixture.offload(hit, dependency))
    fixture.evict([11, 22])
    two_cpu_hits = fixture.request([11, 22], [64, 64])
    assert two_cpu_hits[2][5].tolist() == [64, 64], two_cpu_hits[2][5]
    fixture.check(two_cpu_hits)
    checks.append("merged_two_cpu_hits")

    # A non-page-aligned suffix is computed locally. CPU storage publishes only
    # complete pages, so the next restore must retain/recompute the right suffix.
    partial = fixture.request([11], [79])
    assert partial[2][1].tolist() == [64]
    dependency = fixture.fill_missing(partial)
    fixture.check(partial)
    fixture.flush(fixture.offload(partial, dependency, explicit_mappings=True))
    fixture.evict([11])
    restore = fixture.request([11], [96])
    assert restore[2][1].tolist() == [64], restore[2][1]
    dependency = fixture.fill_missing(restore)
    fixture.check(restore)
    checks.append("partial_cpu_prefix_and_append")
    fixture.flush(fixture.offload(restore, dependency))
    fixture.evict([11])

    mixed = fixture.request([22, 11, 33], [64, 96, 32])
    assert mixed[2][1].tolist() == [64, 96, 0], mixed[2][1]
    assert mixed[2][3][0].item() == 64
    assert mixed[2][5][1].item() == 96
    dependency = fixture.fill_missing(mixed)
    fixture.check(mixed)
    checks.append("gpu_cpu_miss_mix")

    # Queue writes on a non-default producer stream. Offload must publish only
    # after its event, even when immediate allocation/eviction pressures the pool.
    delayed = fixture.request([44], [96])
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        dependency = fixture.fill_missing(delayed, delay=True)
        ticket = fixture.offload(delayed, dependency)
    if not fixture.owner_loop:
        # The legacy runtime has no completion worker. Finished D2H pages stay
        # pinned until the caller reaps/flushes; allocation does not do that.
        # Keep this explicit instead of weakening the owner-pressure test.
        fixture.flush(ticket)
    pressure = fixture.request([55, 66, 77], [128, 128, 128])
    fixture.fill_missing(pressure)
    fixture.check(pressure)
    fixture.flush(ticket)
    fixture.evict([44])
    restored = fixture.request([44], [96])
    assert restored[2][1].tolist() == [96]
    fixture.check(restored)
    checks.append("producer_event_and_allocator_pressure" if fixture.owner_loop
                  else "producer_event_and_allocator_pressure_with_explicit_flush")

    # Invalid device placement must fail without poisoning the owner/next request.
    try:
        fixture.ops.lookup(torch.tensor([44], device="cuda"), torch.tensor([96]), fixture.anchor)
    except RuntimeError as exc:
        assert "CPU" in str(exc)
    else:
        raise AssertionError("invalid CUDA user_ids were accepted")
    fixture.check(fixture.request([44], [96]))
    checks.append("error_recovery")

    final = fixture.request([88], [64])
    dependency = fixture.fill_missing(final, delay=True)
    fixture.offload(final, dependency)
    # Deliberately omit flush in both modes: shutdown must drain producer and
    # transfer work, even though legacy allocation needs explicit caller reaping.
    fixture.ops.shutdown_runtime(fixture.anchor)
    fixture.ops.shutdown_runtime(fixture.anchor)
    checks.append("shutdown_with_pending_offload_and_idempotent_shutdown")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aoti", action="store_true", help="also compile two small native-op graphs")
    parser.add_argument("--legacy", action="store_true", help="exercise owner_loop:0 for comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Package import must not auto-load a different, installed runtime later
    # when registering fake kernels for the AOTI graphs.
    os.environ.pop("KVCACHE_MANAGER_OPS_LIB_DIR", None)
    os.environ["KVCACHE_MANAGER_OPS_LIB"] = str(args.library.resolve())
    torch.ops.load_library(str(args.library.resolve()))
    assert torch.cuda.is_available(), "CUDA required"
    server = fixture = None
    result = {"owner_loop": not args.legacy, "gpu": torch.cuda.get_device_name(), "checks": []}
    try:
        with tempfile.TemporaryDirectory(prefix="recsys-owner-") as temporary:
            server = start_server(Path(temporary), not args.legacy)
            fixture = PagedFixture(owner_loop=not args.legacy)
            if args.aoti:
                aoti_smoke(fixture, args.output)
                result["checks"].append("aoti_publish_and_gpu_cpu_read")
            result["checks"].extend(lifecycle(fixture))
            result["passed"] = True
    except BaseException as exc:
        result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_errors = []
        if fixture is not None:
            try:
                fixture.ops.shutdown_runtime(fixture.anchor)
            except BaseException as exc:
                cleanup_errors.append(f"runtime shutdown: {exc}")
        if server is not None:
            try:
                server.shutdown()
            except BaseException as exc:
                cleanup_errors.append(f"server shutdown: {exc}")
        succeeded_before_cleanup = result.get("passed", False)
        if cleanup_errors:
            result.update(passed=False, cleanup_errors=cleanup_errors)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if cleanup_errors and succeeded_before_cleanup:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
