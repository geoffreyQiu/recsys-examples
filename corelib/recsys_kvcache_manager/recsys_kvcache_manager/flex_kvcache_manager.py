import math
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, Callable, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .hierarchical_kvcache_manager import (
    SecondaryKVCacheManagerBase,
    SecondaryTaskHandle,
    SecondaryTaskStatus,
    SecondaryWaitResult,
    SecondaryErrorCode,
)
from .kvcache_utils import FlexKVIndexMeta, KVIndexMeta, KVLookupResult
from uuid import uuid4
from .kvcache_metadata import KVCacheMetadata

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

@dataclass
class _FlexKVOnloadHandle:
    # Lightweight compatibility handle for native-style wait_host polling.
    # The authoritative completion state is returned by onboard_wait_kvcache().
    task_key: str
    mode: str
    task_ids: torch.Tensor
    slot_mappings: List[torch.Tensor]
    wait_impl: Optional[Callable[[int], SecondaryWaitResult]] = None

    def wait_host(self, _layer_idx: int) -> None:
        # Keep native-compatible wait_host entry, but route to onboard polling.
        if self.wait_impl is not None:
            wait_result = self.wait_impl(_layer_idx)
            if wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            ):
                raise RuntimeError(
                    f"FlexKV onboard wait_host failed: {wait_result.status.value}, msg={wait_result.message}"
                )
            return None
        return None


class FlexKVStorageManager(SecondaryKVCacheManagerBase):
    def __init__(
        self,
        mode: str = "direct",
        server_addr: str = "",
        server_port: int = 0,
        num_layers: int = 0,
        num_heads: int = 0,
        head_dim: int = 0,
        page_size: int = 0,
        secondary_wait_timeout_ms: int = 0,
        secondary_fail_policy: str = "fail_open",
        num_cpu_blocks: int = 4096,
        num_local_blocks: int = 4096,
        num_tmp_cpu_blocks: int = 256,
        enable_mps: bool = False,
    ) -> None:
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = int(server_port)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.secondary_wait_timeout_ms = int(secondary_wait_timeout_ms)
        self.secondary_fail_policy = secondary_fail_policy
        self.num_cpu_blocks = int(num_cpu_blocks)
        self.num_local_blocks = int(num_local_blocks)
        self.num_tmp_cpu_blocks = int(num_tmp_cpu_blocks)
        self.enable_mps = bool(enable_mps)
        self._gpu_cache_table_list: Optional[List[torch.Tensor]] = None
        self._flex_registered_kv_caches: Optional[List[torch.Tensor]] = None
        self._gpu_register_port: str = os.environ.get(
            "FLEXKV_GPU_REGISTER_PORT",
            "ipc:///tmp/flexkv_server_gpu_register",
        )
        self._registered: bool = False
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._adapter = FlexKVClientAdapter(mode, server_addr, server_port)
        self._client = None
        self._ready = False

    def register_gpu_cache_table(self, cache_table) -> None:
        if isinstance(cache_table, torch.Tensor):
            self._gpu_cache_table_list = [cache_table[idx] for idx in range(int(cache_table.size(0)))]
        elif isinstance(cache_table, list):
            self._gpu_cache_table_list = cache_table
        else:
            raise TypeError(f"Unsupported cache_table type: {type(cache_table)}")

        # Initialize FlexKV client only after GPU cache table is available.
        self._init_client()

        # FlexKV VLLM transfer backend assumes [kv, block, ...] contiguous strides. 
        # Our live cache is [block, kv, ...].
        # so register a contiguous mirror buffer and synchronize at transfer boundaries.
        kv_caches = [layer.permute(1, 0, 2, 3, 4).contiguous() for layer in self._gpu_cache_table_list]
        self._flex_registered_kv_caches = kv_caches
        first = kv_caches[0]
        device_id = int(first.device.index if first.device.index is not None else 0)
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=len(kv_caches),
            num_block=int(first.shape[1]),
            tokens_per_block=int(first.shape[2]),
            num_head=int(first.shape[3]),
            head_size=int(first.shape[4]),
            is_mla=False,
        )
        tp_client = KVTPClient(
            gpu_register_port=self._gpu_register_port,
            dp_client_id=0,
            device_id=device_id,
        )
        tp_client.register_to_server(kv_caches=kv_caches, kv_layout=gpu_layout)
        self._registered = True

        # Client becomes operational only after transfer manager is ready.
        if hasattr(self._client, "is_ready"):
            deadline = time.time() + 45.0
            while not self._client.is_ready() and time.time() <= deadline:
                time.sleep(0.05)
        self._ready = True

    # Sync between runtime layout [block, kv, ...] and FlexKV registered layout
    # [kv, block, ...]. to_registered=True means runtime -> registered.
    def _sync_cache_layout(self, to_registered: bool) -> None:
        if (
            self._gpu_cache_table_list is None
            or self._flex_registered_kv_caches is None
            or len(self._gpu_cache_table_list) != len(self._flex_registered_kv_caches)
        ):
            return
        if to_registered:
            src_layers = self._gpu_cache_table_list
            dst_layers = self._flex_registered_kv_caches
        else:
            src_layers = self._flex_registered_kv_caches
            dst_layers = self._gpu_cache_table_list
        for src_layer, dst_layer in zip(src_layers, dst_layers):
            dst_layer.copy_(src_layer.permute(1, 0, 2, 3, 4), non_blocking=False)

    def _init_client(self) -> None:
        if self._client is not None:
            return
        try:
            from flexkv.kvmanager import KVManager
            from flexkv.common.config import ModelConfig, CacheConfig
        except Exception as e:
            raise RuntimeError(f"FlexKV SDK import failed: {e}") from e
        if "FLEXKV_ENABLE_MPS" not in os.environ:
            os.environ["FLEXKV_ENABLE_MPS"] = "1" if self.enable_mps else "0"
        model_cfg = ModelConfig(
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            tp_size=1,
            dp_size=1,
            dtype=torch.bfloat16,
        )
        cache_cfg_kwargs: Dict[str, Any] = {"tokens_per_block": self.page_size}
        # Configure CPU cache sizes if specified.
        if self.num_cpu_blocks > 0:
            cache_cfg_kwargs["num_cpu_blocks"] = self.num_cpu_blocks
        if self.num_local_blocks > 0:
            cache_cfg_kwargs["num_local_blocks"] = self.num_local_blocks
        if self.num_tmp_cpu_blocks > 0:
            cache_cfg_kwargs["num_tmp_cpu_blocks"] = self.num_tmp_cpu_blocks
        cache_cfg = CacheConfig(**cache_cfg_kwargs)
        self._client = KVManager(
            model_config=model_cfg,
            cache_config=cache_cfg,
            dp_client_id=0,
        )
        self._client.start()
    
    #remove if necessary
    # - Applies timeout when secondary_wait_timeout_ms > 0.
    # - Maps FlexKV wait responses into SecondaryTaskStatus.
    def _wait_and_map_result(
        self,
        task_ids: List[int],
        user_ids: List[int],
        timeout_code: str,
        failed_code: str,
    ) -> SecondaryWaitResult:
        try:
            if not task_ids:
                responses = {}
            else:
                wait_kwargs: Dict[str, Any] = {"completely": True}
                if self.secondary_wait_timeout_ms > 0:
                    wait_kwargs["timeout"] = float(self.secondary_wait_timeout_ms) / 1000.0
                responses = self._client.wait(task_ids, **wait_kwargs)
        except Exception as e:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=failed_code,
                message=f"wait exception: {e}",
                failed_user_ids=user_ids,
            )
        if responses is None:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=failed_code,
                message="wait returned None",
                failed_user_ids=user_ids,
            )
        if len(responses) == 0:
            # try_wait() may return empty when tasks are still pending.
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.LAUNCHED,
                ready=False,
                message="pending",
            )
        has_timeout = False
        has_cancelled = False
        has_failed = False
        msgs: List[str] = []
        for tid, resp in responses.items():
            status_name = str(getattr(getattr(resp, "status", None), "name", "UNKNOWN"))
            msgs.append(f"{tid}:{status_name}")
            if status_name == "SUCCESS":
                continue
            if status_name == "TIMEOUT":
                has_timeout = True
            elif status_name == "CANCELLED":
                has_cancelled = True
            else:
                has_failed = True
        if has_timeout:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.TIMEOUT,
                ready=False,
                error_code=timeout_code,
                message=";".join(msgs),
                failed_user_ids=user_ids,
            )
        if has_failed or has_cancelled:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.CANCELLED if has_cancelled and not has_failed else SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=failed_code,
                message=";".join(msgs),
                failed_user_ids=user_ids,
            )
        return SecondaryWaitResult(status=SecondaryTaskStatus.READY, ready=True)

    def _resolve_task_key(self, handle: Any) -> Optional[str]:
        # Fast path: offload now passes task_key as plain string.
        if isinstance(handle, str):
            return handle
        # Backward-compat for legacy tests/callers using object or dict handles.
        if isinstance(handle, dict):
            return handle.get("task_key")
        return getattr(handle, "task_key", None)

    def build_index_meta(
        self,
        user_ids: torch.Tensor,
        history_sequence_lengths: torch.Tensor,
    ) -> FlexKVIndexMeta:
        user_ids_t = user_ids.to(torch.int64)
        seq_lengths_t = history_sequence_lengths.to(device=user_ids_t.device, dtype=torch.int32)
        bsz = int(user_ids_t.size(0))
        zeros_t = torch.zeros((bsz,), dtype=torch.int32, device=user_ids_t.device)
        max_len = int(seq_lengths_t.max().item()) if bsz > 0 else 0
        if max_len > 0:
            token_positions = torch.arange(max_len, dtype=torch.int64, device=user_ids_t.device)
            token_ids = token_positions.unsqueeze(0).expand(bsz, -1).contiguous()
            token_mask = (
                torch.arange(max_len, dtype=torch.int32, device=user_ids_t.device).unsqueeze(0)
                < seq_lengths_t.unsqueeze(1)
            )
        else:
            token_ids = torch.zeros((bsz, 0), dtype=torch.int64, device=user_ids_t.device)
            token_mask = torch.zeros((bsz, 0), dtype=torch.bool, device=user_ids_t.device)
        namespaces = [f"uid:{int(uid)}" for uid in user_ids_t.detach().cpu().tolist()]
        return FlexKVIndexMeta(
            user_ids=user_ids_t,
            seq_lengths=seq_lengths_t,
            batch_size=bsz,
            request_id=str(uuid4()),
            token_ids=token_ids,
            token_mask=token_mask,
            namespaces=namespaces,
            old_cached_lengths=zeros_t,
        )

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        device = index_meta.user_ids.device
        request_id = getattr(index_meta, "request_id", "")

        if getattr(index_meta, "namespaces", None) is None:
            index_meta.namespaces = [f"uid:{int(uid)}" for uid in index_meta.user_ids.detach().cpu().tolist()]

        requests = self._adapter.to_get_match_requests(index_meta)
        task_ids: List[int] = []
        matched_lengths: List[int] = []
        hit_masks: List[np.ndarray] = []
        for req in requests:
            if req["token_ids"].size == 0:
                task_ids.append(-1)
                matched_lengths.append(0)
                hit_masks.append(np.zeros_like(req["token_mask"], dtype=np.bool_))
                continue
            task_id, matched_mask = self._client.get_match(
                token_ids=req["token_ids"],
                token_mask=req["token_mask"],
                namespace=req["namespace"],
            )
            matched_mask = np.asarray(matched_mask, dtype=np.bool_)
            task_ids.append(int(task_id))
            matched_lengths.append(int(matched_mask.sum()))
            hit_masks.append(matched_mask)

        parsed = self._adapter.from_get_match_responses(
            index_meta=index_meta,
            task_ids=task_ids,
            matched_lengths=matched_lengths,
            hit_masks=hit_masks,
        )
        task_ids_t = torch.as_tensor(parsed["task_ids"], dtype=torch.int64, device=device)
        matched_t = torch.as_tensor(parsed["matched_lengths"], dtype=torch.int32, device=device)
        hit_mask_t = parsed.get("hit_mask")
        if isinstance(hit_mask_t, torch.Tensor):
            hit_mask_t = hit_mask_t.to(device=device, dtype=torch.bool)
        else:
            hit_mask_t = None

        index_meta.secondary_get_task_ids = task_ids_t

        return KVLookupResult(
            request_id=request_id,
            user_ids=index_meta.user_ids,
            host_cached_start_indices=torch.zeros_like(matched_t),
            host_cached_lengths=matched_t,
            extra={
                "backend": "flexkv",
                "task_ids": task_ids_t,
                "matched_lengths": matched_t,
                "hit_mask": hit_mask_t,
            },
        )

    def _build_onboard_slot_mappings(self, kvcache_metadata: KVCacheMetadata) -> List[torch.Tensor]:
        mappings: List[torch.Tensor] = []
        kv_indices = kvcache_metadata.kv_indices.to(torch.int64)
        kv_indptr = kvcache_metadata.kv_indptr.to(torch.int64)
        for i in range(int(kv_indptr.numel()) - 1):
            p0 = int(kv_indptr[i].item())
            p1 = int(kv_indptr[i + 1].item())
            page_ids = kv_indices[p0:p1]
            if page_ids.numel() == 0:
                mappings.append(torch.empty((0,), dtype=torch.int64, device=kv_indices.device))
                continue
            # page_ids -> token slots:
            # cat([arange(pid * page_size, (pid + 1) * page_size) for pid in page_ids])
            token_offsets = torch.arange(self.page_size, dtype=torch.int64, device=page_ids.device)
            slot_mapping = (page_ids.unsqueeze(1) * self.page_size + token_offsets.unsqueeze(0)).reshape(-1)
            mappings.append(slot_mapping)
        return mappings

    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> SecondaryTaskHandle:
        task_ids_t = lookup_result.extra["task_ids"].to(
            device=index_meta.user_ids.device, dtype=torch.int64
        )
        kvcache_metadata.onboard_task_ids = task_ids_t
        kvcache_metadata.onboard_slot_mappings = self._build_onboard_slot_mappings(kvcache_metadata)
        launched_slot_mappings = [
            slot_mapping.to(dtype=torch.int64).contiguous().detach().cpu()
            for slot_mapping in kvcache_metadata.onboard_slot_mappings
        ]
        launch_task_ids = [int(tid) for tid in task_ids_t.tolist()]
        launch_user_ids = [int(uid) for uid in index_meta.user_ids.detach().cpu().tolist()]
        task_key = f"onboard:{uuid4()}"
        onload_handle = _FlexKVOnloadHandle(
            task_key=task_key,
            mode=self.mode,
            task_ids=task_ids_t,
            slot_mappings=launched_slot_mappings,
            wait_impl=None,
        )
        onload_task_handle = SecondaryTaskHandle(
            backend="flexkv",
            user_ids=index_meta.user_ids,
            handle=onload_handle,
            status=SecondaryTaskStatus.LAUNCHED,
        )
        onload_handle.wait_impl = lambda _layer_idx: self.onboard_wait_kvcache(onload_task_handle)
        fail_close = str(self.secondary_fail_policy).strip().lower() == "fail_close"

        try:
            launched_task_ids = [int(tid) for tid in self._client.launch(launch_task_ids, launched_slot_mappings)]
            onload_handle.task_ids = torch.as_tensor(
                launched_task_ids, dtype=torch.int64, device=index_meta.user_ids.device
            )
            self._tasks[task_key] = {
                "task_ids": launched_task_ids,
                "user_ids": launch_user_ids,
            }
            return onload_task_handle
        except Exception as e:
            if not fail_close:
                return SecondaryTaskHandle(
                    backend="flexkv",
                    user_ids=index_meta.user_ids,
                    handle=onload_handle,
                    status=SecondaryTaskStatus.SKIPPED,
                    metadata={"reason": f"onboard launch fallback: {e}"},
                )
            return SecondaryTaskHandle(
                backend="flexkv",
                user_ids=index_meta.user_ids,
                handle=onload_handle,
                status=SecondaryTaskStatus.FAILED,
                metadata={"error": f"onboard launch failed: {e}"},
            )

    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        task_key = self._resolve_task_key(task_handle.handle)
        if task_key is None:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.SKIPPED,
                ready=False,
                message="onboard task key missing",
            )
        state = self._tasks.get(task_key)
        if state is None:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.SKIPPED,
                ready=False,
                message="onboard task not launched",
            )
        task_ids = [int(tid) for tid in state.get("task_ids", [])]
        if len(task_ids) == 0:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.SKIPPED,
                ready=False,
                message="no onboard task ids",
            )
        wait_user_ids = [int(uid) for uid in state.get("user_ids", [])]
        wait_result = self._wait_and_map_result(
            task_ids=task_ids,
            user_ids=wait_user_ids,
            timeout_code=SecondaryErrorCode.ONBOARD_TIMEOUT.value,
            failed_code=SecondaryErrorCode.ONBOARD_WAIT_FAILED.value,
        )
        if wait_result.status == SecondaryTaskStatus.READY:
            self._sync_cache_layout(to_registered=False)
            self._tasks.pop(task_key, None)
        return wait_result

    def offload_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: KVCacheMetadata,
    ) -> SecondaryTaskHandle:
        slot_mappings = self._build_onboard_slot_mappings(kvcache_metadata)
        kvcache_metadata.onboard_slot_mappings = slot_mappings
        reqs = self._adapter.to_offload_requests(
            index_meta=index_meta,
            slot_mappings=slot_mappings,
        )
        # FlexKV reads from registered GPU mirror buffers on D2H.
        self._sync_cache_layout(to_registered=True)
        task_ids: List[int] = []
        req_user_ids: List[int] = []
        for req in reqs:
            task_id = self._client.put_async(
                token_ids=req["token_ids"],
                slot_mapping=req["slot_mapping"],
                token_mask=req["token_mask"],
                namespace=req["namespace"],
            )
            task_ids.append(int(task_id))
            req_user_ids.append(int(req["user_id"]))
        task_key = f"offload:{uuid4()}"
        self._tasks[task_key] = {
            "task_ids": task_ids,
            "user_ids": req_user_ids,
        }
        return SecondaryTaskHandle(
            backend="flexkv",
            user_ids=index_meta.user_ids,
            handle=task_key,
            status=SecondaryTaskStatus.LAUNCHED,
        )
     
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        task_key = self._resolve_task_key(task_handle.handle)
        if task_key is None:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=SecondaryErrorCode.OFFLOAD_TASK_NOT_FOUND.value,
                message="offload task key missing",
            )
        state = self._tasks.get(task_key)
        if state is None:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=SecondaryErrorCode.OFFLOAD_TASK_NOT_FOUND.value,
                message="offload task not found",
            )

        task_ids = [int(tid) for tid in state.get("task_ids", [])]
        user_ids = [int(uid) for uid in state.get("user_ids", [])]
        wait_result = self._wait_and_map_result(
            task_ids=task_ids,
            user_ids=user_ids,
            timeout_code=SecondaryErrorCode.OFFLOAD_TIMEOUT.value,
            failed_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
        )
        if wait_result.status in (
            SecondaryTaskStatus.READY,
            SecondaryTaskStatus.TIMEOUT,
            SecondaryTaskStatus.FAILED,
            SecondaryTaskStatus.CANCELLED,
        ):
            self._tasks.pop(task_key, None)
        return wait_result

    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        if task_handle is None or task_handle.handle is None:
            return
        task_key = self._resolve_task_key(task_handle.handle)
        if task_key is None:
            return
        state = self._tasks.get(task_key)
        if state is None:
            return
        task_ids = list(state.get("task_ids", []))
        if len(task_ids) > 0:
            try:
                self._client.cancel(task_ids=task_ids)
            except Exception:
                pass
        self._tasks.pop(task_key, None)
        task_handle.status = SecondaryTaskStatus.CANCELLED
    
    def evict(self, user_ids: torch.Tensor):
        warnings.warn(
            "FlexKV backend does not expose an explicit evict API; evict() is a no-op.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    
    def evict_all(self):
        warnings.warn(
            "FlexKV backend does not expose an explicit evict_all API; evict_all() is a no-op.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    
    def finish_task(self, task_handle: SecondaryTaskHandle) -> bool:
        # FlexKV: wait success -> finish
        return True

class FlexKVClientAdapter:
    def __init__(self, mode: str, server_addr: str = "", server_port: int = 0):
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = server_port

    def to_get_match_requests(self, index_meta: KVIndexMeta) -> List[Dict[str, Any]]:
        token_ids_2d = index_meta.token_ids.to(torch.int64).detach().cpu().numpy()
        token_mask_2d = index_meta.token_mask.to(torch.bool).detach().cpu().numpy()
        reqs: List[Dict[str, Any]] = []
        for i in range(index_meta.batch_size):
            row_ids = token_ids_2d[i]
            row_mask = token_mask_2d[i]
            true_idx = np.where(row_mask)[0]
            if true_idx.size == 0:
                reqs.append(
                    {
                        "user_id": int(index_meta.user_ids[i]),
                        "namespace": [index_meta.namespaces[i]],
                        "token_ids": np.zeros((0,), dtype=np.int64),
                        "token_mask": np.zeros((0,), dtype=np.bool_),
                    }
                )
                continue
            end = int(true_idx[-1]) + 1
            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": [index_meta.namespaces[i]],
                    "token_ids": row_ids[:end].astype(np.int64),
                    "token_mask": row_mask[:end].astype(np.bool_),
                }
            )
        return reqs

    def from_get_match_responses(
        self,
        index_meta: KVIndexMeta,
        task_ids: List[int],
        matched_lengths: List[int],
        hit_masks: List[np.ndarray],
    ) -> Dict[str, Any]:
        batch_size = index_meta.batch_size
        max_len = 0
        if index_meta.token_mask is not None:
            max_len = int(index_meta.token_mask.shape[1])
        elif len(hit_masks) > 0:
            max_len = max(int(m.shape[0]) for m in hit_masks)
        hit_mask_2d = np.zeros((batch_size, max_len), dtype=np.bool_)
        for i, m in enumerate(hit_masks):
            if i >= batch_size:
                break
            upto = min(max_len, int(m.shape[0]))
            hit_mask_2d[i, :upto] = m[:upto]
        return {
            "backend": "flexkv",
            "task_ids": task_ids,
            "matched_lengths": matched_lengths,
            "hit_mask": torch.from_numpy(hit_mask_2d).to(torch.bool) if max_len > 0 else None,
        }

    def to_onboard_launch_payload(
        self,
        index_meta: KVIndexMeta,
        restore_slot_mapping: Any,
    ) -> Dict[str, Any]:
        if not isinstance(restore_slot_mapping, dict):
            return {"task_ids": [], "slot_mappings": []}
        task_ids = [int(x) for x in restore_slot_mapping.get("task_ids", [])]
        slot_mappings = [
            np.asarray(x, dtype=np.int64) for x in restore_slot_mapping.get("slot_mappings", [])
        ]
        valid_task_ids: List[int] = []
        valid_slot_mappings: List[np.ndarray] = []
        for t, s in zip(task_ids, slot_mappings):
            if s.ndim != 1 or s.size == 0:
                continue
            valid_task_ids.append(t)
            valid_slot_mappings.append(s)
        return {"task_ids": valid_task_ids, "slot_mappings": valid_slot_mappings}

    def to_offload_requests(
        self,
        index_meta: KVIndexMeta,
        slot_mappings: List[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        token_ids_2d = index_meta.token_ids.to(torch.int64).detach().cpu().numpy()
        seq_lengths = index_meta.seq_lengths.to(torch.int64).detach().cpu().tolist()
        namespaces = index_meta.namespaces


        reqs: List[Dict[str, Any]] = []
        num_items = min(index_meta.batch_size, len(slot_mappings))
        for i in range(num_items):
            slot_mapping_tensor = slot_mappings[i]
            if slot_mapping_tensor is None or slot_mapping_tensor.numel() == 0:
                continue
            slot_mapping_full = slot_mapping_tensor.to(torch.int64).detach().cpu().numpy()

            row_ids = token_ids_2d[i]
            valid_len = min(int(seq_lengths[i]), int(row_ids.shape[0]))
            if valid_len <= 0:
                continue
            req_len = int(slot_mapping_full.size)
            valid_token_ids = row_ids[:valid_len].astype(np.int64)
            if valid_len >= req_len:
                req_token_ids = valid_token_ids[:req_len]
            else:
                # Pad to full block-aligned length so FlexKV cache_engine won't drop tail tokens.
                pad_value = int(valid_token_ids[-1])
                pad_tokens = np.full((req_len - valid_len,), pad_value, dtype=np.int64)
                req_token_ids = np.concatenate([valid_token_ids, pad_tokens], axis=0)

            if namespaces is not None and i < len(namespaces):
                namespace = namespaces[i]
            else:
                namespace = f"uid:{int(index_meta.user_ids[i])}"

            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": [namespace],
                    "token_ids": req_token_ids,
                    "token_mask": None,
                    "slot_mapping": slot_mapping_full,
                }
            )
        return reqs
