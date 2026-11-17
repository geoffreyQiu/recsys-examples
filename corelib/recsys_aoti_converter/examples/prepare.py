# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic package locally; no weights or inputs are distributed."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from toy_attention import ToyAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    shutil.copy2(Path(__file__).with_name("toy_attention.py"), source)
    parameters = dict(vocabulary=37, embedding_dim=24, query_dim=12, heads=2, head_dim=64)
    torch.manual_seed(73)
    model = ToyAttention(**parameters).eval()
    torch.save(model.state_dict(), source / "weights.pt")
    (source / "hyperparameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    np.savez(args.output / "sample.npz",
             history_ids=torch.randint(0, 37, (4, 16)).numpy(),
             queries=torch.randn(4, 3, 12).numpy())
    descriptor = {
        "source": "source", "class": "toy_attention:ToyAttention",
        "input_names": "toy_attention:INPUT_NAMES",
        "hyperparameters": "hyperparameters.json", "checkpoint": "weights.pt",
        "sample": {"format": "npz", "path": "sample.npz"},
    }
    (args.output / "model.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    spec = {"history_input_indices": [0], "length": 16, "heads": 2,
            "head_dim": 64, "policy": "immutable_snapshot"}
    (args.output / "cache.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
