import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
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


@dataclass
class _FlexKVOnloadHandle:
    task_key: str
    request_id: str
    mode: str
    task_ids: torch.Tensor
    slot_mappings: List[torch.Tensor]
    phase: str = "onboard"

    def wait_host(self, _layer_idx: int) -> None:
        # FlexKV onboard path does not expose layer-wise native event handles.
        return None

    # Compatibility helpers for legacy dict-like access patterns.
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@dataclass
class _FlexKVOffloadHandle:
    task_key: str
    task_ids: torch.Tensor
    kind: str
    mode: str
    request_id: str

    # Compatibility helpers for legacy dict-like access patterns.
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


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
        self._gpu_register_port: str = os.environ.get(
            "FLEXKV_GPU_REGISTER_PORT",
            "ipc:///tmp/flexkv_server_gpu_register",
        )
        self._registered: bool = False
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._adapter = FlexKVClientAdapter(mode, server_addr, server_port)
        self._client = None
        self._ready = False

    def _to_numpy_2d(self, tensor: Optional[torch.Tensor], np_dtype):
        if tensor is None:
            return None
        arr = tensor.detach().cpu().numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {arr.shape}")
        return arr.astype(np_dtype, copy=False)

    def register_gpu_cache_table(self, cache_table) -> None:
        if isinstance(cache_table, torch.Tensor):
            self._gpu_cache_table_list = [cache_table[idx] for idx in range(int(cache_table.size(0)))]
        elif isinstance(cache_table, list):
            self._gpu_cache_table_list = cache_table
        else:
            raise TypeError(f"Unsupported cache_table type: {type(cache_table)}")

        if self._client is not None and not self._registered:
            self._register_gpu_cache_tensors()

    def _register_gpu_cache_tensors(self) -> None:
        if self._registered:
            return
        if self._gpu_cache_table_list is None or len(self._gpu_cache_table_list) == 0:
            return
        try:
            from flexkv.server.client import KVTPClient
            from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
        except Exception as e:
            raise RuntimeError(f"FlexKV GPU registration import failed: {e}") from e

        kv_caches = [layer.permute(1, 0, 2, 3, 4).contiguous() for layer in self._gpu_cache_table_list]
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

    def _ensure_client_ready(self) -> None:
        if self._ready and self._client is not None:
            return
        try:
            from flexkv.kvmanager import KVManager
            from flexkv.common.config import ModelConfig, CacheConfig
        except Exception as e:
            raise RuntimeError(f"FlexKV SDK import failed: {e}") from e
        if "FLEXKV_ENABLE_MPS" not in os.environ:
            os.environ["FLEXKV_ENABLE_MPS"] = "1" if self.enable_mps else "0"
        if self._client is None:
            model_cfg = ModelConfig(
                num_layers=self.num_layers,
                num_kv_heads=self.num_heads,
                head_size=self.head_dim,
                tp_size=1,
                dp_size=1,
                dtype=torch.bfloat16,
            )
            cache_cfg_kwargs: Dict[str, Any] = {"tokens_per_block": self.page_size}
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
            self._register_gpu_cache_tensors()
        elif not self._registered:
            self._register_gpu_cache_tensors()

        # Client becomes operational only after transfer manager is ready.
        if hasattr(self._client, "is_ready"):
            deadline = time.time() + 45.0
            while not self._client.is_ready() and time.time() <= deadline:
                time.sleep(0.05)
        self._ready = True
    def _wait_task_ids(self, task_ids: List[int]) -> Dict[int, Any]:
        if len(task_ids) == 0:
            return {}
        if self.secondary_wait_timeout_ms > 0:
            timeout_s = float(self.secondary_wait_timeout_ms) / 1000.0
            return self._client.wait(task_ids, timeout=timeout_s, completely=True)
        return self._client.try_wait(task_ids)
    def _convert_wait_result(
        self,
        responses: Dict[int, Any],
        user_ids: List[int],
        timeout_code: str,
        failed_code: str,
    ) -> SecondaryWaitResult:
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

    def build_index_meta(
        self,
        user_ids: torch.Tensor,
        history_sequence_lengths: torch.Tensor,
    ) -> FlexKVIndexMeta:
        user_ids_t = user_ids.to(torch.int64)
        seq_lengths_t = history_sequence_lengths.to(device=user_ids_t.device, dtype=torch.int32)
        bsz = int(user_ids_t.size(0))
        zeros_t = torch.zeros((bsz,), dtype=torch.int32, device=user_ids_t.device)
        namespaces = [f"uid:{int(uid)}" for uid in user_ids_t.detach().cpu().tolist()]
        return FlexKVIndexMeta(
            user_ids=user_ids_t,
            seq_lengths=seq_lengths_t,
            batch_size=bsz,
            request_id=str(uuid4()),
            namespaces=namespaces,
            old_cached_lengths=zeros_t,
        )

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        bsz = int(index_meta.batch_size)
        device = index_meta.user_ids.device
        request_id = getattr(index_meta, "request_id", "")
        zeros_i32 = torch.zeros((bsz,), dtype=torch.int32, device=device)
        zeros_i64 = torch.zeros((bsz,), dtype=torch.int64, device=device)

        def _empty_lookup(error_code: Optional[str] = None, error_msg: str = "") -> KVLookupResult:
            index_meta.secondary_get_task_ids = zeros_i64
            index_meta.secondary_matched_lengths = zeros_i32
            index_meta.secondary_hit_mask = None
            index_meta.old_cached_lengths = zeros_i32
            extra: Dict[str, Any] = {
                "backend": "flexkv",
                "task_ids": zeros_i64,
                "matched_lengths": zeros_i32,
                "hit_mask": None,
            }
            if error_code is not None:
                extra["error_code"] = error_code
            if error_msg:
                extra["error"] = error_msg
            return KVLookupResult(
                request_id=request_id,
                user_ids=index_meta.user_ids,
                host_cached_start_indices=zeros_i32,
                host_cached_lengths=zeros_i32,
                extra=extra,
            )

        if index_meta.token_ids is None or index_meta.token_mask is None:
            return _empty_lookup(SecondaryErrorCode.LOOKUP_MISSING_TOKENS.value)

        if getattr(index_meta, "namespaces", None) is None:
            index_meta.namespaces = [f"uid:{int(uid)}" for uid in index_meta.user_ids.detach().cpu().tolist()]

        try:
            self._ensure_client_ready()
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
            index_meta.secondary_matched_lengths = matched_t
            index_meta.secondary_hit_mask = hit_mask_t
            index_meta.old_cached_lengths = matched_t

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
        except Exception as e:
            print(f"[WARN] FlexKV lookup_kvcache failed: {e}")
            return _empty_lookup(
                error_code=SecondaryErrorCode.LOOKUP_FAILED.value,
                error_msg=str(e),
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
            # page_id -> token slot base
            slot_mapping = torch.repeat_interleave(page_ids * self.page_size, repeats=self.page_size)
            mappings.append(slot_mapping)
        return mappings

    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata,
    ) -> SecondaryTaskHandle:
        task_ids = (lookup_result.extra or {}).get("task_ids")
        if task_ids is None:
            task_ids_t = torch.empty((0,), dtype=torch.int64, device=index_meta.user_ids.device)
        elif isinstance(task_ids, torch.Tensor):
            task_ids_t = task_ids.to(device=index_meta.user_ids.device, dtype=torch.int64)
        else:
            task_ids_t = torch.as_tensor(task_ids, dtype=torch.int64, device=index_meta.user_ids.device)

        kvcache_metadata.onboard_task_ids = task_ids_t
        kvcache_metadata.onboard_slot_mappings = self._build_onboard_slot_mappings(kvcache_metadata)

        request_id = getattr(index_meta, "request_id", "")
        return SecondaryTaskHandle(
            backend="flexkv",
            user_ids=index_meta.user_ids,
            handle=_FlexKVOnloadHandle(
                task_key=f"onboard:{request_id}",
                request_id=request_id,
                mode=self.mode,
                task_ids=task_ids_t,
                slot_mappings=kvcache_metadata.onboard_slot_mappings,
            ),
            status=SecondaryTaskStatus.LAUNCHED,
        )

    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY,
            ready=True,
            message="flexkv onboard ready",
        )

    def offload_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor] = None,
    ) -> SecondaryTaskHandle:
        reqs = self._adapter.to_offload_requests(
            index_meta=index_meta,
            append_slot_mapping=append_slot_mapping,
            append_slot_indptr=append_slot_indptr,
            tokens_per_block=self.page_size,
        )
        if len(reqs) == 0:
            return SecondaryTaskHandle(
                backend="flexkv",
                handle=None,
                status=SecondaryTaskStatus.SKIPPED,
                metadata={"reason": "no offload blocks"},
            )
        self._ensure_client_ready()
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
        request_id = getattr(index_meta, "request_id", "")
        task_key = f"offload:{request_id}"
        self._tasks[task_key] = {
            "task_ids": task_ids,
            "kind": "offload",
            "user_ids": req_user_ids,
        }
        return SecondaryTaskHandle(
            backend="flexkv",
            user_ids=index_meta.user_ids,
            handle=_FlexKVOffloadHandle(
                task_key=task_key,
                task_ids=torch.as_tensor(task_ids, dtype=torch.int64, device=index_meta.user_ids.device),
                kind="offload",
                mode=self.mode,
                request_id=request_id,
            ),
            status=SecondaryTaskStatus.LAUNCHED,
            metadata={"request_id": request_id},
        )
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        if task_handle is None or task_handle.handle is None:
            return SecondaryWaitResult(status=SecondaryTaskStatus.SKIPPED, ready=True)
        try:
            task_key = task_handle.handle.get("task_key")
            state = self._tasks.get(task_key)
            if state is None:
                return SecondaryWaitResult(
                    status=SecondaryTaskStatus.FAILED,
                    ready=False,
                    error_code=SecondaryErrorCode.OFFLOAD_TASK_NOT_FOUND.value,
                    message="offload task not found",
                )
            responses = self._wait_task_ids(list(state.get("task_ids", [])))
            return self._convert_wait_result(
                responses=responses,
                user_ids=list(state.get("user_ids", [])),
                timeout_code=SecondaryErrorCode.OFFLOAD_TIMEOUT.value,
                failed_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
            )
        except Exception as e:
            return SecondaryWaitResult(
                status=SecondaryTaskStatus.FAILED,
                ready=False,
                error_code=SecondaryErrorCode.OFFLOAD_WAIT_FAILED.value,
                message=f"offload_wait exception: {e}",
            )
    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        if task_handle is None or task_handle.handle is None:
            return
        task_key = task_handle.handle.get("task_key")
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
        return None
    def evict_all(self):
        return None
    def finish_task(self, task_handle: SecondaryTaskHandle) -> bool:
        # FlexKV 场景下 wait 成功即视为完成
        return True

class FlexKVClientAdapter:
    def __init__(self, mode: str, server_addr: str = "", server_port: int = 0):
        self.mode = mode
        self.server_addr = server_addr
        self.server_port = server_port

    def _to_numpy_2d(self, tensor: Optional[torch.Tensor], np_dtype) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or None, got {type(tensor)}")
        arr = tensor.detach().cpu().numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape={arr.shape}")
        return arr.astype(np_dtype, copy=False)

    def to_get_match_requests(self, index_meta: KVIndexMeta) -> List[Dict[str, Any]]:
        token_ids_2d = self._to_numpy_2d(index_meta.token_ids, np.int64)
        token_mask_2d = self._to_numpy_2d(index_meta.token_mask, np.bool_)
        if token_ids_2d is None or token_mask_2d is None:
            return []
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
            "restore_slot_mapping": None,
            "append_slot_mapping": None,
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
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor],
        tokens_per_block: int,
    ) -> List[Dict[str, Any]]:
        if append_slot_mapping is None or append_slot_indptr is None:
            return []
        if index_meta.token_ids is None or index_meta.token_mask is None:
            return []

        page_ids = append_slot_mapping.to(torch.int64).detach().cpu().numpy()
        indptr = append_slot_indptr.to(torch.int64).detach().cpu().numpy()
        token_ids_2d = index_meta.token_ids.to(torch.int64).detach().cpu().numpy()
        token_mask_2d = index_meta.token_mask.to(torch.bool).detach().cpu().numpy()
        if index_meta.secondary_matched_lengths is None:
            matched_lengths = []
        else:
            matched_lengths = (
                index_meta.secondary_matched_lengths
                .to(torch.int64)
                .detach()
                .cpu()
                .tolist()
            )


        reqs: List[Dict[str, Any]] = []
        for i in range(index_meta.batch_size):
            p0 = int(indptr[i])
            p1 = int(indptr[i + 1])
            if p1 <= p0:
                continue
            req_pages = page_ids[p0:p1]
            slot_mapping_full = np.repeat(req_pages * tokens_per_block, tokens_per_block).astype(np.int64)

            row_ids = token_ids_2d[i]
            row_mask = token_mask_2d[i]
            valid_token_ids = row_ids[row_mask]

            old_cached = int(index_meta.old_cached_lengths[i])
            if i < len(matched_lengths):
                old_cached = max(old_cached, int(matched_lengths[i]))

            append_token_ids = valid_token_ids[old_cached:]
        
            aligned = (append_token_ids.size // tokens_per_block) * tokens_per_block
            aligned = min(aligned, slot_mapping_full.size)
            if aligned <= 0:
                continue

            reqs.append(
                {
                    "user_id": int(index_meta.user_ids[i]),
                    "namespace": [index_meta.namespaces[i]],
                    "token_ids": append_token_ids[:aligned].astype(np.int64),
                    "token_mask": np.ones((aligned,), dtype=np.bool_),
                    "slot_mapping": slot_mapping_full[:aligned].astype(np.int64),
                }
            )
        return reqs
