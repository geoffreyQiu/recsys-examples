import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml  # type: ignore[import-untyped]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KVCACHE_CONFIG_FILE = SCRIPT_DIR / "kvcache_cpp_runtime.yaml"


def resolve_kvcache_config_file(config_file: Optional[str] = None) -> Path:
    path = config_file or os.environ.get("KVCACHE_MANAGER_CONFIG_FILE")
    return Path(path).resolve() if path else DEFAULT_KVCACHE_CONFIG_FILE.resolve()


def load_kvcache_runtime_yaml(
    config_file: Optional[str] = None,
) -> tuple[Dict[str, Any], Path]:
    path = resolve_kvcache_config_file(config_file)
    with path.open("r", encoding="utf-8") as config_stream:
        config = yaml.safe_load(config_stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"KV-cache config must be a YAML mapping: {path}")
    return config, path


def prepare_flexkv_runtime_yaml(
    config_file: Optional[str] = None,
) -> tuple[Dict[str, Any], Path]:
    """Select task cleanup for the installed adapter without editing user YAML.

    The pinned adapter predating RankInfo retains child records after batched
    completion. Newer adapters use the default aggregate-task policy. An
    explicit YAML setting takes precedence over this compatibility default.
    """
    from flexkv.common import config as flexkv_config

    config, path = load_kvcache_runtime_yaml(config_file)
    if "onboard_child_tasks" not in config and not hasattr(flexkv_config, "RankInfo"):
        config["onboard_child_tasks"] = 1
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="recsys-kvcache-runtime-", suffix=".yaml", delete=False
        ) as derived:
            yaml.safe_dump(config, derived, sort_keys=False)
            path = Path(derived.name)
    return config, path


def required_config(config: Dict[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required KV-cache config value: {name}")
    return value


def config_int(config: Dict[str, Any], name: str) -> int:
    try:
        return int(required_config(config, name))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got {config.get(name)!r}"
        ) from exc


def config_optional_int(config: Dict[str, Any], name: str) -> Optional[int]:
    value = config.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def config_float(config: Dict[str, Any], name: str, default: float) -> float:
    value = config.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def config_bool(config: Dict[str, Any], name: str) -> Optional[bool]:
    value = config.get(name)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def config_torch_dtype(config: Dict[str, Any]) -> torch.dtype:
    value = str(required_config(config, "dtype")).strip().lower()
    if value in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    if value in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    raise ValueError(f"dtype must be bfloat16 or float16, got {value!r}")


def ipc_socket_path(endpoint: str) -> Optional[Path]:
    prefix = "ipc://"
    if not endpoint.startswith(prefix):
        return None
    return Path(endpoint[len(prefix) :])


def reset_ipc_socket(endpoint: str) -> None:
    socket_path = ipc_socket_path(endpoint)
    if socket_path is None:
        return
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass


def extra_flexkv_configs(config: Dict[str, Any]) -> Dict[str, Any]:
    configs: Dict[str, Any] = {}
    for name in ("enable_p2p_cpu", "enable_p2p_ssd", "enable_3rd_remote"):
        value = config_bool(config, name)
        if value is not None:
            configs[name] = value
    for name in ("redis_host", "local_ip", "redis_password"):
        value = config.get(name)
        if value:
            configs[name] = str(value)
    redis_port = config_optional_int(config, "redis_port")
    if redis_port is not None:
        configs["redis_port"] = redis_port
    return configs


def make_flexkv_server_configs(kvcache_config, cpu_cache_gb: int, ssd_cache_gb: int):
    """Build config for the pinned binary-protocol adapter or newer FlexKV."""
    from flexkv.common import config as flexkv_config

    model = flexkv_config.ModelConfig(
        num_layers=kvcache_config.num_layers,
        num_kv_heads=kvcache_config.num_heads,
        head_size=kvcache_config.head_dim,
        dtype=kvcache_config.dtype,
        tp_size=1,
        dp_size=1,
    )
    extras = kvcache_config.extra_configs or {}
    if hasattr(flexkv_config, "RankInfo"):
        cache = flexkv_config.CacheConfig(tokens_per_block=kvcache_config.page_size)
        user = flexkv_config.UserConfig()
        user.cpu_cache_gb = cpu_cache_gb
        user.ssd_cache_gb = ssd_cache_gb
        for name, value in extras.items():
            setattr(user, name, value)
        flexkv_config.update_default_config_from_user_config(
            flexkv_config.RankInfo(model_config=model), cache, user
        )
    else:
        # FlexKV 1.0's binary-protocol adapter uses explicit block counts.
        # Keep its CPU-only startup usable without importing newer helpers.
        if ssd_cache_gb or any(extras.get(key) for key in (
            "enable_p2p_cpu", "enable_p2p_ssd", "enable_3rd_remote"
        )):
            raise ValueError("This FlexKV adapter needs a newer config API for SSD/P2P/remote tiers")
        element_bytes = torch.empty((), dtype=kvcache_config.dtype).element_size()
        block_bytes = (kvcache_config.num_layers * 2 * kvcache_config.page_size
                       * kvcache_config.num_heads * kvcache_config.head_dim * element_bytes)
        blocks = max(1, int(cpu_cache_gb * 1024**3 // block_bytes))
        cache = flexkv_config.CacheConfig(
            tokens_per_block=kvcache_config.page_size,
            num_cpu_blocks=blocks, num_local_blocks=blocks,
            num_tmp_cpu_blocks=max(16, kvcache_config.max_batch_size),
            enable_ssd=False, enable_p2p_cpu=False, enable_p2p_ssd=False,
        )
    return model, cache
