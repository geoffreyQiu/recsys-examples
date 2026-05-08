import math
import os
import time

import numpy as np
import torch
# from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .gpu_kvcache_manager import GPUKVCacheManager
from .hierarchical_kvcache_manager import (
    SecondaryKVCacheManagerBase,
    SecondaryTaskHandle,
    SecondaryTaskStatus,
    SecondaryWaitResult,
    SecondaryErrorCode,
)
from .native_host_kvcache_manager import (
    NativeHostKVCacheManager
)
from .flex_kvcache_manager import FlexKVStorageManager
# from .hierarchical_kvcache_manager import NopSecondaryKVCacheManager
from .kvcache_config import KVCacheConfig
from .kvcache_utils import KVCacheOffloadMode, KVLookupResult, KVIndexMeta
from .kvcache_metadata import KVCacheMetadata


class KVCacheManager:
    def __init__(
        self,
        gpu_kvcache_manager: GPUKVCacheManager,
        secondary_kvcache_manager: Optional[SecondaryKVCacheManagerBase] = None,
        offload_mode: str = "lazy",
        # secondary_fail_policy: str = "fail_open",
    ):
        # self.num_layers = num_layers
        # self.num_heads = num_kv_heads
        # self.head_dim = kv_headdim
        # self.page_size = num_tokens_per_page
        # self.chunk_size = num_tokens_per_chunk
        # self.num_primary_cache_pages = num_primary_cache_pages
        # self.num_buffer_pages = num_buffer_pages
        # self.max_batch_size = max_batch_size

        self.gpu_kvcache_mgr = gpu_kvcache_manager
        self.dummy_empty_tensor = torch.tensor([], dtype=torch.int32)

        self.secondary_kvcache_manager = secondary_kvcache_manager # if secondary_kvcache_manager is not None else NopSecondaryKVCacheManager()
        self.secondary_kvcache_manager.register_gpu_cache_table(self.gpu_kvcache_mgr.gpu_kvcache_table)

        self.offload_mode = (
            KVCacheOffloadMode(offload_mode)
            if offload_mode in {m.value for m in KVCacheOffloadMode}
            else KVCacheOffloadMode.LAZY
        )
        # self.secondary_fail_policy = secondary_fail_policy
        self.ongoing_onboard_tasks: List[SecondaryTaskHandle] = []
        self.ongoing_offload_tasks: List[SecondaryTaskHandle] = []
        # FlexKV fallback token cache:
        # keep full tokens by uid so lookup can recover tokens
        # when caller does not pass token_ids/token_mask.
        self._flex_lookup_token_cache: Dict[int, Dict[str, Any]] = {}

    def _cache_flex_lookup_tokens(self, index_meta: KVIndexMeta) -> None:
        if not isinstance(self.secondary_kvcache_manager, FlexKVStorageManager):
            return
        token_ids = getattr(index_meta, "token_ids", None)
        if not isinstance(token_ids, torch.Tensor):
            return

        seq_lengths = index_meta.seq_lengths.to(torch.int64).detach().cpu()
        token_ids_cpu = token_ids.to(torch.int64).detach().cpu()
        token_mask = getattr(index_meta, "token_mask", None)
        token_mask_cpu = (
            token_mask.to(torch.bool).detach().cpu()
            if isinstance(token_mask, torch.Tensor)
            else None
        )
        namespaces = getattr(index_meta, "namespaces", None)

        for i, uid_t in enumerate(index_meta.user_ids.detach().cpu().tolist()):
            uid = int(uid_t)
            if i >= token_ids_cpu.shape[0]:
                continue
            row_ids = token_ids_cpu[i]
            valid_len = min(int(seq_lengths[i].item()), int(row_ids.numel()))
            if token_mask_cpu is not None and i < token_mask_cpu.shape[0]:
                row_mask = token_mask_cpu[i]
                true_idx = torch.where(row_mask)[0]
                if true_idx.numel() == 0:
                    valid_len = 0
                else:
                    valid_len = min(valid_len, int(true_idx[-1].item()) + 1)
            cached_ids = row_ids[:valid_len].clone()
            namespace = (
                str(namespaces[i])
                if isinstance(namespaces, list) and i < len(namespaces)
                else f"uid:{uid}"
            )
            self._flex_lookup_token_cache[uid] = {
                "token_ids": cached_ids,
                "namespace": namespace,
            }

    def _recover_flex_lookup_tokens(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[str]]]:
        if not isinstance(self.secondary_kvcache_manager, FlexKVStorageManager):
            return None, None, None

        rows: List[torch.Tensor] = []
        valid_lens: List[int] = []
        namespaces: List[str] = []
        for uid_t, seq_len_t in zip(user_ids.detach().cpu().tolist(), sequence_lengths.detach().cpu().tolist()):
            uid = int(uid_t)
            state = self._flex_lookup_token_cache.get(uid)
            if state is None:
                return None, None, None
            cached_ids = state.get("token_ids")
            if not isinstance(cached_ids, torch.Tensor):
                return None, None, None
            seq_len = int(seq_len_t)
            valid_len = min(seq_len, int(cached_ids.numel()))
            rows.append(cached_ids[:valid_len])
            valid_lens.append(valid_len)
            namespaces.append(str(state.get("namespace", f"uid:{uid}")))

        batch_size = int(user_ids.numel())
        max_len = max(valid_lens) if len(valid_lens) > 0 else 0
        device = user_ids.device
        token_ids = torch.zeros((batch_size, max_len), dtype=torch.int64, device=device)
        token_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for i, (row_ids, valid_len) in enumerate(zip(rows, valid_lens)):
            if valid_len <= 0:
                continue
            token_ids[i, :valid_len] = row_ids[:valid_len].to(device=device, dtype=torch.int64)
            token_mask[i, :valid_len] = True
        return token_ids, token_mask, namespaces
    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        gpu_lookup_results = self.gpu_kvcache_mgr.lookup(user_ids)
        index_meta = self.secondary_kvcache_manager.build_index_meta(user_ids, sequence_lengths)
        if token_ids is None and token_mask is None:
            recovered_ids, recovered_mask, recovered_namespaces = self._recover_flex_lookup_tokens(
                user_ids=user_ids,
                sequence_lengths=sequence_lengths,
            )
            if recovered_ids is not None and recovered_mask is not None:
                token_ids = recovered_ids
                token_mask = recovered_mask
                if namespaces is None:
                    namespaces = recovered_namespaces
        if token_ids is not None and hasattr(index_meta, "token_ids"):
            index_meta.token_ids = token_ids.to(device=user_ids.device, dtype=torch.int64)
        if token_mask is not None and hasattr(index_meta, "token_mask"):
            index_meta.token_mask = token_mask.to(device=user_ids.device, dtype=torch.bool)
        if namespaces is not None and hasattr(index_meta, "namespaces"):
            if len(namespaces) != int(user_ids.numel()):
                raise ValueError(
                    f"namespaces size mismatch: got {len(namespaces)}, expected {int(user_ids.numel())}"
                )
            index_meta.namespaces = list(namespaces)

        host_lookup_results: KVLookupResult
        # For FlexKV, keep index_meta.token_ids full-length while narrowing host lookup mask:
        # - gpu_start > 0 and gpu_len > 0: lookup [0, gpu_start)
        # - gpu_start == 0 and gpu_len > 0: skip host lookup
        # - gpu_len == 0: lookup [0, seq_len)
        if (
            isinstance(self.secondary_kvcache_manager, FlexKVStorageManager)
            and hasattr(index_meta, "token_ids")
            and isinstance(getattr(index_meta, "token_ids", None), torch.Tensor)
            and isinstance(getattr(index_meta, "token_mask", None), torch.Tensor)
            and gpu_lookup_results.gpu_cached_start_indices is not None
            and gpu_lookup_results.gpu_cached_lengths is not None
        ):
            full_mask = index_meta.token_mask.to(device=user_ids.device, dtype=torch.bool)
            max_len = int(full_mask.shape[1])
            gpu_start = gpu_lookup_results.gpu_cached_start_indices.to(device=user_ids.device, dtype=torch.int32)
            gpu_len = gpu_lookup_results.gpu_cached_lengths.to(device=user_ids.device, dtype=torch.int32)
            seq_len_i32 = sequence_lengths.to(device=user_ids.device, dtype=torch.int32)
            host_target_len = torch.where(gpu_len > 0, gpu_start, seq_len_i32)
            host_target_len = torch.clamp(host_target_len, min=0, max=max_len)

            positions = torch.arange(max_len, device=user_ids.device, dtype=torch.int32).unsqueeze(0)
            host_mask = (positions < host_target_len.unsqueeze(1)) & full_mask

            # Build a temporary index_meta used only for secondary lookup,
            # so returned index_meta keeps full token mask for offload path.
            host_index_meta = self.secondary_kvcache_manager.build_index_meta(user_ids, sequence_lengths)
            if hasattr(host_index_meta, "token_ids"):
                host_index_meta.token_ids = index_meta.token_ids
            if hasattr(host_index_meta, "token_mask"):
                host_index_meta.token_mask = host_mask
            if hasattr(host_index_meta, "namespaces"):
                host_index_meta.namespaces = getattr(index_meta, "namespaces", None)
            if hasattr(host_index_meta, "request_id"):
                host_index_meta.request_id = getattr(index_meta, "request_id", "")
            host_lookup_results = self.secondary_kvcache_manager.lookup_kvcache(host_index_meta)
        else:
            host_lookup_results = self.secondary_kvcache_manager.lookup_kvcache(index_meta)

        lookup_results = KVLookupResult.merge(gpu_lookup_results, host_lookup_results)
        return index_meta, lookup_results

    def allocate_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None
    ) -> KVCacheMetadata:
        return self.gpu_kvcache_mgr.allocate(
            index_meta.user_ids,
            index_meta.seq_lengths,
            lookup_results,
            output_kvcache_metadata=output_kvcache_metadata,
        )
    
    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata
    ) -> SecondaryTaskHandle:
        task_handle = self.secondary_kvcache_manager.onboard_launch_kvcache(
            index_meta, lookup_result, kvcache_metadata,
        )
        kvcache_metadata.kv_onload_handle = task_handle

        # Skip recording the ongoing onboard tasks. For now there should be only one task.
        # self.ongoing_onboard_tasks.append(task_handle)
        return task_handle
    
    def onboard_try_wait_kvcache_or_fail(
        self,
        kv_index_meta: KVIndexMeta,
        task_handle: Optional[SecondaryTaskHandle],
    ) -> Optional[SecondaryWaitResult]:
        if task_handle is None:
            return SecondaryWaitResult(status=SecondaryTaskStatus.READY, ready=True)
        wait_result = self.secondary_kvcache_manager.onboard_wait_kvcache(task_handle)
        if wait_result.status in (
            SecondaryTaskStatus.FAILED,
            SecondaryTaskStatus.TIMEOUT,
            SecondaryTaskStatus.CANCELLED,
        ):
            if task_handle.backend == "native":
                # Native backend supports page-level revoke on onboard failure.
                metadata = task_handle.metadata or {}
                if (
                    "onboard_page_starts" in metadata
                    and "num_onboard_pages" in metadata
                ):
                    self.gpu_kvcache_mgr.revoke_onboard_pages(
                        kv_index_meta.user_ids,
                        metadata["onboard_page_starts"],
                        metadata["num_onboard_pages"],
                    )
            elif task_handle.backend == "flexkv":
                # FlexKV does not expose native-style onboard page revoke metadata.
                # In fail_close mode, evict to avoid serving potentially inconsistent cache.
                fail_policy = str(
                    getattr(self.secondary_kvcache_manager, "secondary_fail_policy", "fail_open")
                ).strip().lower()
                if fail_policy == "fail_close":
                    self.evict(kv_index_meta.user_ids, for_gpu=True, for_host=True)
        return wait_result

    
    def _offload_kvcache_impl(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        if isinstance(self.secondary_kvcache_manager, FlexKVStorageManager):
            if kvcache_metadata is None:
                raise ValueError("flexkv offload requires kvcache_metadata")
            # Keep full tokens for future lookup calls that don't explicitly pass token_ids.
            self._cache_flex_lookup_tokens(index_meta)
            task_handle = self.secondary_kvcache_manager.offload_launch_kvcache(
                index_meta=index_meta,
                kvcache_metadata=kvcache_metadata,
            )
            if task_handle is not None and task_handle.status == SecondaryTaskStatus.LAUNCHED:
                self.ongoing_offload_tasks.append(task_handle)
            return task_handle

        # 1. Get the maximum batch for offloading from GPU
        uids_to_offload = self.gpu_kvcache_mgr.check_for_offload(index_meta.user_ids)
        print(f"[DEBUG] Offload uids: {uids_to_offload}")

        # 2. Lookup host again in case of multi-GPU instances
        offloaded_lengths = self.secondary_kvcache_manager.lookup_kvcache(index_meta).host_cached_lengths
        print(f"[DEBUG] Offloaded len (another host lookup): {offloaded_lengths}")

        # 3. Acquire and lock GPU cache pages
        (offload_user_ids, 
         offload_start_indices, 
         offload_page_indices_list, 
        ) = self.gpu_kvcache_mgr.acquire_offload_pages(uids_to_offload, offloaded_lengths)
        # returned with cache pages locked (per-uid). TODO(junyiq): per-(uid, lock_start, lock_end)
        # Note: all pages from offload_page_indicesa are required to offload. duplication will be resolved when finished.
        if offload_user_ids.size(0) == 0:
            return None
        print(f"[DEBUG] Offload uids: {offload_user_ids}")
        print(f"[DEBUG] Offload startpos: {offload_start_indices}")
        print(f"[DEBUG] Offload lens: {offload_page_indices_list}")

        # 4. Launch the offloading thru Host
        task_handle = self.secondary_kvcache_manager.offload_launch_kvcache(
            offload_user_ids, offload_start_indices, offload_page_indices_list,
        )
        if task_handle is None:
            self.gpu_kvcache_mgr.release_offload_pages(
                offload_user_ids, offload_start_indices, self.dummy_empty_tensor,
                offloaded=False)
            return None
        
        self.ongoing_offload_tasks.append(task_handle)
        return task_handle
    
    def eager_offboard_kvcache(
        self,
        index_meta: KVIndexMeta,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.EAGER:
            return None
        return self._offload_kvcache_impl(index_meta)


    def lazy_offload_kvcache(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.LAZY:
            return None
        return self._offload_kvcache_impl(index_meta, kvcache_metadata)
        

    def finish_or_cancel_kvcache_ops(self) -> None:
        remain_tasks = []
        for task_handle in self.ongoing_offload_tasks:
            wait_result = self.secondary_kvcache_manager.offload_wait_kvcache(task_handle)
            if wait_result.status == SecondaryTaskStatus.LAUNCHED:
                remain_tasks.append(task_handle)
                continue
            if wait_result.status == SecondaryTaskStatus.READY:
                offload_success = self.secondary_kvcache_manager.finish_task(task_handle)
            elif wait_result.status in (SecondaryTaskStatus.SKIPPED, SecondaryTaskStatus.EVENT_READY):
                offload_success = False
            elif wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            ):
                offload_success = False
                self.secondary_kvcache_manager.cancel_task(task_handle)
            else:
                raise RuntimeError(
                    f"Unexpected offload wait result status: {wait_result.status.value}, msg={wait_result.message}"
                )
            if task_handle.backend != "flexkv":
                self.gpu_kvcache_mgr.release_offload_pages(
                    *(self.secondary_kvcache_manager.get_offload_handle_metadata(task_handle)),
                    offloaded=offload_success,
                )
        self.ongoing_offload_tasks = remain_tasks
    

    def evict(self, user_ids: torch.Tensor, for_gpu: bool = False, for_host: bool = False):
        if for_gpu:
            self.gpu_kvcache_mgr.evict(user_ids)
        if for_host:
            self.secondary_kvcache_manager.evict(user_ids)

    def evict_all(self, for_gpu: bool = False, for_host: bool = False):
        if for_gpu:
            self.gpu_kvcache_mgr.evict_all()
        if for_host:
            self.secondary_kvcache_manager.evict_all()
    
    @staticmethod
    def _build_secondary_manager_from_config(kvcache_config) -> SecondaryKVCacheManagerBase:
        if kvcache_config.secondary_backend == "native":
            from .native_host_kvcache_manager import NativeHostKVCacheManager
            return NativeHostKVCacheManager(
                kvcache_config.num_layers,
                kvcache_config.num_heads,
                kvcache_config.head_dim,
                kvcache_config.page_size,
                kvcache_config.offload_chunksize,
                kvcache_config.host_capacity_per_layer,
                kvcache_config.max_batch_size,
                math.ceil(
                    kvcache_config.max_seq_len / kvcache_config.page_size
                ) * kvcache_config.page_size,
                kvcache_config.onload_timeout_ms,
                kvcache_config.offload_timeout_ms,
                kvcache_config.dtype,
                kvcache_config.device,
            )
        elif kvcache_config.secondary_backend == "flexkv":
            extra = getattr(kvcache_config, "extra_configs", {}) or {}
            flexkv_mode = extra.get("flexkv_mode", "direct")
            flexkv_server_addr = extra.get("flexkv_server_addr", "")
            flexkv_server_port = int(extra.get("flexkv_server_port", 0))
            flexkv_num_cpu_blocks = int(extra.get("flexkv_num_cpu_blocks", 4096))
            flexkv_num_local_blocks = int(extra.get("flexkv_num_local_blocks", 4096))
            flexkv_num_tmp_cpu_blocks = int(extra.get("flexkv_num_tmp_cpu_blocks", 256))
            flexkv_secondary_fail_policy = str(extra.get("flexkv_secondary_fail_policy", "fail_open"))
            flexkv_enable_mps_raw = extra.get("flexkv_enable_mps", 0)
            if isinstance(flexkv_enable_mps_raw, str):
                flexkv_enable_mps = flexkv_enable_mps_raw.strip().lower() in {"1", "true", "yes", "on"}
            else:
                flexkv_enable_mps = bool(flexkv_enable_mps_raw)
            
            return FlexKVStorageManager(
                mode=flexkv_mode,
                server_addr=flexkv_server_addr,
                server_port=flexkv_server_port,
                num_layers=kvcache_config.num_layers,
                num_heads=kvcache_config.num_heads,
                head_dim=kvcache_config.head_dim,
                page_size=kvcache_config.page_size,
                secondary_wait_timeout_ms=int(kvcache_config.offload_timeout_ms),
                secondary_fail_policy=flexkv_secondary_fail_policy,
                num_cpu_blocks=flexkv_num_cpu_blocks,
                num_local_blocks=flexkv_num_local_blocks,
                num_tmp_cpu_blocks=flexkv_num_tmp_cpu_blocks,
                enable_mps=flexkv_enable_mps,
                # secondary_wait_timeout_ms=secondary_wait_timeout_ms,
                # secondary_fail_policy=secondary_fail_policy,
            )
        else:
            print(f"Unknown host kvcache backend {kvcache_config.secondary_backend}")
        
        # return NopSecondaryKVCacheManager()
        return None
        

    @classmethod
    def from_config(cls, kvcache_config):

        gpu_kvcache_mgr = GPUKVCacheManager(
            kvcache_config.num_layers,
            kvcache_config.num_heads,
            kvcache_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.offload_chunksize,
            kvcache_config.num_primary_cache_pages,
            kvcache_config.num_buffer_pages,
            kvcache_config.max_batch_size,
            math.ceil(
                kvcache_config.max_seq_len / kvcache_config.page_size
            ) * kvcache_config.page_size,
            kvcache_config.dtype,
            kvcache_config.device,
        )
        
        host_kvcache_mgr = cls._build_secondary_manager_from_config(
            kvcache_config
        )

        return cls(
            gpu_kvcache_mgr,
            host_kvcache_mgr,
            kvcache_config.offload_mode,
            # kvcache_config.offload_timeout_ms,
            # kvcache_config.secondary_fail_policy,
        )
