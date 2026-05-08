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
            # Note: Either evict the total user or revoke affected pages
            self.gpu_kvcache_mgr.revoke_onboard_pages(
                kv_index_meta.user_ids,
                task_handle.metadata["onboard_page_starts"],
                task_handle.metadata["num_onboard_pages"],
            )
            # if self.secondary_fail_policy == "fail_close":
            #     self.gpu_kvcache_mgr.evict(kv_index_meta.user_ids)
            #     raise RuntimeError(
            #         f"onboard wait failed: status={wait_result.status.value}, msg={wait_result.message}"
            #     )
        return wait_result

    
    def _build_flex_append_mapping(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: KVCacheMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_indices = kvcache_metadata.kv_indices.to(torch.int64)
        kv_indptr = kvcache_metadata.kv_indptr.to(torch.int64)
        seq_lengths = index_meta.seq_lengths.to(torch.int64)
        old_cached = getattr(index_meta, "old_cached_lengths", None)
        if old_cached is None:
            old_cached = torch.zeros_like(seq_lengths, dtype=torch.int64)
        else:
            old_cached = old_cached.to(device=seq_lengths.device, dtype=torch.int64)
        page_size = int(self.gpu_kvcache_mgr.page_size)
        chunks: List[torch.Tensor] = []
        indptr: List[int] = [0]
        for i in range(int(index_meta.user_ids.numel())):
            s = int(kv_indptr[i].item())
            e = int(kv_indptr[i + 1].item())
            seq_pages = kv_indices[s:e]
            start_blk = int(old_cached[i].item()) // page_size
            end_blk = (int(seq_lengths[i].item()) + page_size - 1) // page_size
            append_pages = seq_pages[start_blk:end_blk]
            chunks.append(append_pages)
            indptr.append(indptr[-1] + int(append_pages.numel()))
        append_slot_mapping = (
            torch.cat(chunks, dim=0)
            if len(chunks) > 0
            else torch.empty((0,), dtype=torch.int64, device=kv_indices.device)
        )
        append_slot_indptr = torch.tensor(indptr, dtype=torch.int64, device=kv_indices.device)
        return append_slot_mapping, append_slot_indptr

    def _offload_kvcache_impl(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: Optional[KVCacheMetadata] = None,
    ):
        if isinstance(self.secondary_kvcache_manager, FlexKVStorageManager):
            if kvcache_metadata is None:
                raise ValueError("flexkv offload requires kvcache_metadata")
            append_slot_mapping, append_slot_indptr = self._build_flex_append_mapping(
                index_meta=index_meta,
                kvcache_metadata=kvcache_metadata,
            )
            task_handle = self.secondary_kvcache_manager.offload_launch_kvcache(
                index_meta=index_meta,
                append_slot_mapping=append_slot_mapping,
                append_slot_indptr=append_slot_indptr,
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
