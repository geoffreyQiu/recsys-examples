# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nve_aoti_export_round_trip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    hstu_root = repo_root / "examples/hstu"
    if str(hstu_root) not in sys.path:
        sys.path.insert(0, str(hstu_root))
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
    pynve = pytest.importorskip("pynve")
    from inference_aoti.nve_aoti_compat import load_aoti
    from pynve.torch.nve_export import export_aot
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
        metadata = json.load(metadata_file)
    if pynve.__version__.startswith("26.05."):
        assert isinstance(metadata, list)
        assert metadata and all("cache_type" in layer for layer in metadata)
    else:
        assert metadata["version"] == 2
        assert metadata["layers"]

    session = load_aoti(export_dir, device=device)
    assert session.num_layers == 1

    with torch.inference_mode():
        outputs = session.run([keys, offsets])
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], reference)
    session.close()
