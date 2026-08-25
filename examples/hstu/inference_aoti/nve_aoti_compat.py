# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-safe loading for NVE-backed AOTInductor packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence


class NveCompatibilityError(RuntimeError):
    """Raised when an NVE runtime or artifact has an unsupported contract."""


class OutputDirectoryError(RuntimeError):
    """Raised when exporter output paths could merge or overwrite data."""


def prepare_output_directories(
    export_dir: str | os.PathLike[str],
    dump_dir: str | os.PathLike[str],
) -> tuple[str, str]:
    """Validate both exporter destinations before creating either one."""
    if not os.fspath(export_dir) or not os.fspath(dump_dir):
        raise OutputDirectoryError("--export_dir and --dump_dir must not be empty")

    resolved = {
        "export_dir": Path(export_dir).resolve(),
        "dump_dir": Path(dump_dir).resolve(),
    }
    export_path = resolved["export_dir"]
    dump_path = resolved["dump_dir"]
    if (
        export_path == dump_path
        or export_path in dump_path.parents
        or dump_path in export_path.parents
    ):
        raise OutputDirectoryError(
            "--export_dir and --dump_dir must be distinct, non-nested paths"
        )

    for argument, path in resolved.items():
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise OutputDirectoryError(
                f"--{argument} must be absent or an empty directory; refusing "
                f"to merge stale output at {path}"
            )
    for path in resolved.values():
        path.mkdir(parents=True, exist_ok=True)
    return str(export_path), str(dump_path)


def _classify_metadata(metadata: Any) -> str:
    if isinstance(metadata, list):
        if not metadata:
            raise NveCompatibilityError(
                "NVE artifact metadata error: legacy layer list is empty"
            )
        for index, layer in enumerate(metadata):
            if not isinstance(layer, dict):
                raise NveCompatibilityError(
                    f"NVE artifact metadata error: legacy layer {index} is not an object"
                )
            if "cache_type" not in layer or "layer_type" in layer:
                raise NveCompatibilityError(
                    "NVE artifact metadata error: legacy layers must contain "
                    "cache_type and must not contain layer_type"
                )
        return "26.05"

    if isinstance(metadata, dict):
        version = metadata.get("version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != 2
        ):
            raise NveCompatibilityError(
                f"NVE artifact metadata error: unsupported schema version {version!r}"
            )
        if not isinstance(metadata.get("resources"), dict):
            raise NveCompatibilityError(
                "NVE artifact metadata error: schema v2 resources must be an object"
            )
        layers = metadata.get("layers")
        if not isinstance(layers, list) or not layers:
            raise NveCompatibilityError(
                "NVE artifact metadata error: schema v2 layers must be a non-empty list"
            )
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise NveCompatibilityError(
                    f"NVE artifact metadata error: schema v2 layer {index} is not an object"
                )
            if "layer_type" not in layer or "cache_type" in layer:
                raise NveCompatibilityError(
                    "NVE artifact metadata error: schema v2 layers must contain "
                    "layer_type and must not contain cache_type"
                )
        return "26.07"

    raise NveCompatibilityError(
        "NVE artifact metadata error: expected a legacy array or schema-v2 object"
    )


def _artifact_generation(package_dir: str | os.PathLike[str]) -> str:
    metadata_path = Path(package_dir) / "metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except FileNotFoundError as error:
        raise NveCompatibilityError(
            f"NVE artifact metadata error: missing {metadata_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise NveCompatibilityError(
            f"NVE artifact metadata error: cannot read {metadata_path}: {error}"
        ) from error
    return _classify_metadata(metadata)


def _runtime_generation() -> str:
    import pynve

    version = str(getattr(pynve, "__version__", "unknown"))
    generation = next(
        (
            candidate
            for candidate in ("26.05", "26.07")
            if version == candidate or version.startswith(f"{candidate}.")
        ),
        None,
    )
    if generation is None:
        raise NveCompatibilityError(
            f"Unsupported pynve version {version!r}; expected 26.05.x or 26.07.x"
        )

    package_file = getattr(pynve, "__file__", None)
    if package_file is None:
        raise NveCompatibilityError("The imported pynve package has no __file__")
    resolved_file = Path(package_file).resolve()
    expected_prefix = Path(f"/opt/nve/{generation}/python").resolve()
    if not resolved_file.is_relative_to(expected_prefix):
        raise NveCompatibilityError(
            f"pynve {version} was imported from {resolved_file}; expected it below "
            f"{expected_prefix}"
        )
    return generation


def _normalize_outputs(outputs: Any) -> list["torch.Tensor"]:
    import torch

    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (tuple, list)) and all(
        isinstance(output, torch.Tensor) for output in outputs
    ):
        return list(outputs)
    raise TypeError(
        "AOTI runner returned an unsupported result; expected a tensor, tuple, "
        f"or list of tensors, got {type(outputs)!r}"
    )


class AotiSession:
    """Own an AOTI loader and the NVE layers on which it depends."""

    def __init__(self, loader: Any, layers: Sequence[Any]) -> None:
        self._loader = loader
        self._layers = list(layers)

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def run(self, inputs: Sequence["torch.Tensor"]) -> list["torch.Tensor"]:
        if self._loader is None:
            raise RuntimeError("AOTI session is closed")
        if hasattr(self._loader, "run"):
            outputs = self._loader.run(list(inputs))
        else:
            outputs = self._loader(tuple(inputs))
        return _normalize_outputs(outputs)

    def close(self) -> None:
        if self._loader is None:
            return
        loader = self._loader
        self._loader = None
        del loader
        self._layers.clear()

    def __enter__(self) -> "AotiSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_aoti(
    package_dir: str | os.PathLike[str],
    device: "torch.device",
) -> AotiSession:
    """Load an AOTI package only with its matching selected NVE runtime."""
    package_dir = os.fspath(package_dir)
    artifact_generation = _artifact_generation(package_dir)
    runtime_generation = _runtime_generation()
    if runtime_generation != artifact_generation:
        raise NveCompatibilityError(
            "NVE version mismatch: selected runtime "
            f"{runtime_generation}, artifact requires {artifact_generation}"
        )

    if runtime_generation == "26.07":
        from pynve.torch.nve_export import load_aot

        loader, layers = load_aot(package_dir, device=device)
        return AotiSession(loader, layers)

    import torch
    from pynve.torch.nve_export import load_nve_layers

    if device.type != "cuda":
        raise NveCompatibilityError("NVE 26.05 AOTI loading requires a CUDA device")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    with torch.cuda.device(device_index):
        layers = load_nve_layers(package_dir)
    from torch._C._aoti import AOTIModelPackageLoader

    loader = AOTIModelPackageLoader(
        os.path.join(package_dir, "model.pt2"),
        "model",
        False,
        1,
        device_index,
    )
    return AotiSession(loader, layers)
