# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nve_aoti_export_round_trip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ops_dir = Path(
        os.environ.get(
            "DYNAMICEMB_OPS_LIB_DIR",
            repo_root / "corelib/dynamicemb/torch_binding_build",
        )
    )
    torch.ops.load_library(str(ops_dir / "inference_emb_ops.so"))

    # Register export fakes after the corresponding custom operators exist.
    import dynamicemb.index_range_meta  # noqa: F401
    import dynamicemb.lookup_meta  # noqa: F401
    from dynamicemb.exportable_tables import (
        create_inference_embedding_collection,
    )
    from pynve.torch.nve_export import export_aot, load_aot
    from torchrec.modules.embedding_configs import EmbeddingConfig

    device = torch.device("cuda", torch.cuda.current_device())
    model = create_inference_embedding_collection(
        [
            EmbeddingConfig(
                name="table",
                num_embeddings=8,
                embedding_dim=4,
                feature_names=["feature"],
            )
        ],
        pooling_mode=-1,
        use_dynamic=False,
    ).eval()
    model.load_from_embedding_table(
        torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4)
    )

    keys = torch.tensor([0, 1, 3, 7], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, keys.numel()], dtype=torch.int64, device=device)
    with torch.inference_mode():
        reference = model(keys, offsets)

    export_dir = tmp_path / "nve_aoti_model"
    export_aot(model, (keys, offsets), str(export_dir))

    with (export_dir / "metadata.json").open() as metadata_file:
        assert json.load(metadata_file)["version"] == 2

    compiled_loader, loaded_nve_layers = load_aot(export_dir, device=device)
    assert len(loaded_nve_layers) == 1

    with torch.inference_mode():
        outputs = compiled_loader.run([keys, offsets])
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], reference)
