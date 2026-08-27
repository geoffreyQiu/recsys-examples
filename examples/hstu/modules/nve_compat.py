# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small compatibility helpers for the supported NVE Python generations."""

import os
import site
from pathlib import Path


_DEFAULT_NVE_INSTALL_ROOT = Path("/opt/nve")
_SUPPORTED_GENERATIONS = ("26.05", "26.06", "26.07")


def _expected_pynve_dir(generation: str) -> Path:
    configured = os.environ.get(
        "NVE_INSTALL_ROOT", os.fspath(_DEFAULT_NVE_INSTALL_ROOT)
    )
    if not configured:
        raise RuntimeError("NVE_INSTALL_ROOT must not be empty")

    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise RuntimeError(
            f"NVE_INSTALL_ROOT must be an absolute path, got {configured!r}"
        )
    install_root = root.resolve()
    site_roots = (
        Path(site_root).expanduser().resolve()
        for site_root in (*site.getsitepackages(), site.getusersitepackages())
        if site_root
    )
    containing_site_root = next(
        (
            site_root
            for site_root in site_roots
            if install_root.is_relative_to(site_root)
        ),
        None,
    )
    if containing_site_root is not None:
        raise RuntimeError(
            f"NVE_INSTALL_ROOT={install_root} is inside Python's automatically "
            f"searched site-packages root {containing_site_root}. Use an isolated "
            "root such as /opt/nve"
        )
    return (install_root / generation / "python" / "pynve").resolve()


def imported_nve_generation() -> str:
    """Return the generation of the already selected ``pynve`` package.

    The development image deliberately has no unversioned pynve installation.
    A selected package must come from
    ``NVE_INSTALL_ROOT/<generation>/python``.
    """
    import pynve

    version = str(getattr(pynve, "__version__", "unknown"))
    generation = next(
        (
            candidate
            for candidate in _SUPPORTED_GENERATIONS
            if version == candidate or version.startswith(f"{candidate}.")
        ),
        None,
    )
    if generation is None:
        raise RuntimeError(
            f"Unsupported pynve version {version!r}; expected 26.05.x, "
            "26.06.x, or 26.07.x"
        )

    declared_generation = os.environ.get("NVE_VERSION")
    if declared_generation:
        if declared_generation not in _SUPPORTED_GENERATIONS:
            raise RuntimeError(
                f"Unsupported NVE_VERSION={declared_generation!r}; expected "
                "exactly 26.05, 26.06, or 26.07"
            )
        if declared_generation != generation:
            raise RuntimeError(
                f"NVE_VERSION={declared_generation} but imported pynve "
                f"{version} from {getattr(pynve, '__file__', None)}"
            )

    package_file = getattr(pynve, "__file__", None)
    if package_file is None:
        raise RuntimeError("The imported pynve package has no __file__")
    actual_dir = Path(package_file).resolve().parent
    expected_dir = _expected_pynve_dir(generation)
    if actual_dir != expected_dir:
        raise RuntimeError(
            f"pynve {version} was imported from {actual_dir}; expected "
            f"{expected_dir}. Select exactly one versioned NVE Python prefix "
            "through PYTHONPATH"
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
