# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small compatibility helpers for the supported NVE Python generations."""

from pathlib import Path
from typing import Optional


_SUPPORTED_PREFIXES = {
    "26.05": Path("/opt/nve/26.05/python"),
    "26.07": Path("/opt/nve/26.07/python"),
}


def imported_nve_generation() -> str:
    """Return the generation of the already selected ``pynve`` package.

    The development image deliberately has no unversioned pynve installation,
    so a selected package must come from its matching versioned prefix.
    """
    import pynve

    version = str(getattr(pynve, "__version__", "unknown"))
    generation: Optional[str] = next(
        (
            candidate
            for candidate in _SUPPORTED_PREFIXES
            if version == candidate or version.startswith(f"{candidate}.")
        ),
        None,
    )
    if generation is None:
        raise RuntimeError(
            f"Unsupported pynve version {version!r}; expected 26.05.x or 26.07.x"
        )

    package_file = getattr(pynve, "__file__", None)
    if package_file is None:
        raise RuntimeError("The imported pynve package has no __file__")
    resolved_file = Path(package_file).resolve()
    expected_prefix = _SUPPORTED_PREFIXES[generation].resolve()
    if not resolved_file.is_relative_to(expected_prefix):
        raise RuntimeError(
            f"pynve {version} was imported from {resolved_file}; expected it below "
            f"{expected_prefix}"
        )
    return generation


def gpu_only_constructor_kwargs() -> dict[str, object]:
    """Return the generation-specific selector for a GPU-only NVE layer."""
    import pynve.torch.nve_layers as nve_layers

    if imported_nve_generation() == "26.05":
        return {"cache_type": nve_layers.CacheType.NoCache}
    return {"layer_type": nve_layers.LayerType.GPULayer}


def hierarchical_constructor_kwargs(storage: object) -> dict[str, object]:
    """Return the selector and backing store for a hierarchical NVE layer."""
    import pynve.torch.nve_layers as nve_layers

    if imported_nve_generation() == "26.05":
        return {
            "cache_type": nve_layers.CacheType.Hierarchical,
            "remote_interface": storage,
        }
    return {
        "layer_type": nve_layers.LayerType.Hierarchical,
        "storage": storage,
    }


def needs_legacy_embedding_lookup_fake_override() -> bool:
    """Return true only for the selected NVE 26.05 package."""
    try:
        return imported_nve_generation() == "26.05"
    except ModuleNotFoundError:
        return False
