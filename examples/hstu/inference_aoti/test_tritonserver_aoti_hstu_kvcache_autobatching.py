# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import numpy as np
import torch

INPUT_NAMES = ("INPUT__0", "INPUT__1", "INPUT__2", "INPUT__3", "INPUT__4")
INPUT_SUFFIXES = (
    "values",
    "lengths",
    "num_candidates",
    "user_ids",
    "total_history_lengths",
)
OUTPUT_ATOL = 0.0625


def _load_tensor(path: Path) -> np.ndarray:
    return torch.jit.load(str(path), map_location="cpu").tensor.numpy()


def _load_requests(dump_dir: Path, count: int):
    prefix = dump_dir / "batch_000000"
    values, lengths, num_candidates, user_ids, history_lengths = [
        _load_tensor(Path(f"{prefix}_{suffix}.pt")) for suffix in INPUT_SUFFIXES
    ]
    reference = _load_tensor(Path(f"{prefix}_compiled_logits.pt"))

    token_counts = lengths.sum(axis=1, dtype=np.int64)
    offsets = np.pad(np.cumsum(token_counts), (1, 0))
    order = list(
        dict.fromkeys(
            [
                int(token_counts.argmin()),
                int(token_counts.argmax()),
                *range(len(lengths)),
            ]
        )
    )[:count]
    if len(order) != count:
        raise RuntimeError(
            f"The first dump batch contains fewer than {count} requests"
        )
    if len(set(user_ids[order].tolist())) != count:
        raise RuntimeError("Concurrent KV-cache requests must use distinct users")

    requests = []
    for index in order:
        arrays = [
            values[offsets[index] : offsets[index + 1]].reshape(1, -1),
            lengths[index : index + 1],
            num_candidates[index : index + 1].reshape(1, 1),
            user_ids[index : index + 1].reshape(1, 1),
            history_lengths[index : index + 1].reshape(1, 1),
        ]
        requests.append((arrays, reference[index : index + 1]))
    return requests


def _make_inputs(httpclient, arrays):
    inputs = []
    for name, array in zip(INPUT_NAMES, arrays):
        infer_input = httpclient.InferInput(name, array.shape, "INT64")
        infer_input.set_data_from_numpy(array.astype(np.int64, copy=False))
        inputs.append(infer_input)
    return inputs


def _batch_counts(client, model_name: str) -> dict[int, int]:
    stats = client.get_inference_statistics(model_name=model_name)["model_stats"][0]
    return {
        int(item["batch_size"]): int(item["compute_infer"]["count"])
        for item in stats.get("batch_stats", [])
    }


def _batch_delta(before: dict[int, int], after: dict[int, int]) -> dict[int, int]:
    return {
        batch_size: after.get(batch_size, 0) - before.get(batch_size, 0)
        for batch_size in sorted(before.keys() | after.keys())
        if after.get(batch_size, 0) > before.get(batch_size, 0)
    }


def _run_burst(client, httpclient, model_name: str, requests, phase: str) -> float:
    output = [httpclient.InferRequestedOutput("OUTPUT__0")]
    handles = [
        client.async_infer(
            model_name,
            inputs=_make_inputs(httpclient, arrays),
            outputs=output,
            request_id=f"hstu-kv-{phase}-{index}",
        )
        for index, (arrays, _) in enumerate(requests)
    ]
    max_abs_diff = max(
        float(
            np.max(
                np.abs(handle.get_result().as_numpy("OUTPUT__0") - reference)
            )
        )
        for handle, (_, reference) in zip(handles, requests)
    )
    if max_abs_diff > OUTPUT_ATOL:
        raise RuntimeError(f"{phase} output parity failed: {max_abs_diff:.6f}")
    return max_abs_diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="KV-cache Triton autobatching smoke"
    )
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model_name", default="hstu_gr_ranking_kvcache")
    parser.add_argument("--batch_size", type=int, choices=(2, 4, 8), default=8)
    args = parser.parse_args()

    import tritonclient.http as httpclient

    requests = _load_requests(args.dump_dir, args.batch_size)
    client = httpclient.InferenceServerClient(
        url=args.url,
        concurrency=args.batch_size,
    )

    before = _batch_counts(client, args.model_name)
    miss_diff = _run_burst(client, httpclient, args.model_name, requests, "miss")
    after_miss = _batch_counts(client, args.model_name)
    hit_diff = _run_burst(client, httpclient, args.model_name, requests, "hit")
    after_hit = _batch_counts(client, args.model_name)

    miss_batches = _batch_delta(before, after_miss)
    hit_batches = _batch_delta(after_miss, after_hit)
    if not all(
        any(batch_size > 1 for batch_size in counts)
        for counts in (miss_batches, hit_batches)
    ):
        raise RuntimeError("Triton did not dynamically batch both request bursts")

    print(
        "KV-cache autobatching passed: "
        f"miss_batch_stats={miss_batches}, hit_batch_stats={hit_batches}, "
        f"max_abs_diff={max(miss_diff, hit_diff):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
