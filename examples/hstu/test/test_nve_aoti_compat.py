# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import sys
from pathlib import Path

import pytest
import torch

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from inference_aoti import nve_aoti_compat  # noqa: E402


LEGACY_METADATA = [
    {
        "id": 0,
        "module_path": "embedding",
        "cache_type": "LinearUVM",
    }
]
SCHEMA_V2_METADATA = {
    "version": 2,
    "resources": {"mb-1": {"type": "Managed"}},
    "layers": [
        {
            "id": 0,
            "module_path": "embedding",
            "layer_type": "LinearUVM",
        }
    ],
}


def _write_metadata(directory: Path, metadata) -> None:
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (LEGACY_METADATA, "legacy-v1"),
        (SCHEMA_V2_METADATA, "schema-v2"),
    ],
)
def test_artifact_contract(tmp_path: Path, metadata, expected: str) -> None:
    package_dir = tmp_path / expected
    _write_metadata(package_dir, metadata)
    assert nve_aoti_compat._artifact_contract(package_dir) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        [{"layer_type": "LinearUVM"}],
        [{"cache_type": "LinearUVM"}, {"layer_type": "LinearUVM"}],
        {"version": 1, "resources": {}, "layers": [{"layer_type": "LinearUVM"}]},
        {"version": 2.0, "resources": {}, "layers": [{"layer_type": "LinearUVM"}]},
        {"version": True, "resources": {}, "layers": [{"layer_type": "LinearUVM"}]},
        {"version": 2, "layers": [{"layer_type": "LinearUVM"}]},
        {"version": 2, "resources": None, "layers": [{"layer_type": "LinearUVM"}]},
        {"version": 2, "resources": {}, "layers": []},
        {
            "version": 2,
            "resources": {},
            "layers": [{"cache_type": "LinearUVM", "layer_type": "LinearUVM"}],
        },
    ],
)
def test_invalid_metadata_is_rejected(tmp_path: Path, metadata) -> None:
    package_dir = tmp_path / "invalid"
    _write_metadata(package_dir, metadata)
    with pytest.raises(
        nve_aoti_compat.NveCompatibilityError,
        match="NVE artifact metadata error",
    ):
        nve_aoti_compat._artifact_contract(package_dir)


def test_missing_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        nve_aoti_compat.NveCompatibilityError,
        match="missing .*metadata.json",
    ):
        nve_aoti_compat._artifact_contract(tmp_path)


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("26.05", "legacy-v1"),
        ("26.06", "schema-v2"),
        ("26.07", "schema-v2"),
    ],
)
def test_runtime_contract(generation: str, expected: str) -> None:
    assert nve_aoti_compat._runtime_contract(generation) == expected


def test_output_directories_are_created_only_after_both_validate(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "model"
    dump_dir = tmp_path / "replay"
    resolved_export, resolved_dump = nve_aoti_compat.prepare_output_directories(
        export_dir, dump_dir
    )
    assert Path(resolved_export) == export_dir.resolve()
    assert Path(resolved_dump) == dump_dir.resolve()
    assert export_dir.is_dir()
    assert dump_dir.is_dir()


def test_nonempty_output_rejects_before_creating_other_path(tmp_path: Path) -> None:
    export_dir = tmp_path / "model"
    export_dir.mkdir()
    (export_dir / "existing").write_text("stale", encoding="utf-8")
    dump_dir = tmp_path / "replay"

    with pytest.raises(
        nve_aoti_compat.OutputDirectoryError,
        match="--export_dir .* refusing to merge stale output",
    ):
        nve_aoti_compat.prepare_output_directories(export_dir, dump_dir)
    assert not dump_dir.exists()


def test_nested_output_directories_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        nve_aoti_compat.OutputDirectoryError,
        match="distinct, non-nested",
    ):
        nve_aoti_compat.prepare_output_directories(
            tmp_path / "output", tmp_path / "output" / "replay"
        )


def test_runtime_artifact_mismatch_fails_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "legacy"
    _write_metadata(package_dir, LEGACY_METADATA)
    monkeypatch.setattr(nve_aoti_compat, "_runtime_generation", lambda: "26.07")

    with pytest.raises(
        nve_aoti_compat.NveCompatibilityError,
        match=(
            "NVE contract mismatch: selected runtime 26.07 uses schema-v2, "
            "artifact requires legacy-v1"
        ),
    ):
        nve_aoti_compat.load_aoti(package_dir, torch.device("cuda", 0))


@pytest.mark.parametrize("kind", ["tensor", "tuple", "list"])
def test_session_normalizes_outputs(kind: str) -> None:
    output = torch.tensor([1.0])

    class Loader:
        def run(self, inputs):
            if kind == "tensor":
                return output
            if kind == "tuple":
                return (output,)
            return [output]

    session = nve_aoti_compat.AotiSession(Loader(), [object()])
    assert session.run([torch.tensor([0])]) == [output]
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.run([torch.tensor([0])])


def test_session_releases_loader_before_layers() -> None:
    released = []

    class Tracked:
        def __init__(self, name: str) -> None:
            self.name = name

        def __del__(self) -> None:
            released.append(self.name)

    session = nve_aoti_compat.AotiSession(Tracked("loader"), [Tracked("layer")])
    session.close()
    gc.collect()
    assert released == ["loader", "layer"]
