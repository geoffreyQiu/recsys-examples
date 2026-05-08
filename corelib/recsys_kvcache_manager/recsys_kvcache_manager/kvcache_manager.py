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
    SecondaryErrorCode,
    SecondaryTaskStatus,
    SecondaryTaskHandle,
    SecondaryWaitResult,
)
from .native_host_kvcache_manager import (
    NativeHostKVCacheManager
)
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
        # self.ongoing_onboard_tasks: List[str, SecondaryTaskHandle] = list()
        self.ongoing_offload_tasks: List[str, SecondaryTaskHandle] = list()
    
    def lookup_kvcache(
        self,
        user_ids: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> Tuple[KVIndexMeta, KVLookupResult]:
        gpu_lookup_results = self.gpu_kvcache_mgr.lookup(user_ids)
        index_meta = self.secondary_kvcache_manager.build_index_meta(user_ids, sequence_lengths)
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

    
    def _offload_kvcache_impl(
        self,
        index_meta: KVIndexMeta,
    ):
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
            # offload is rejected on the host side (e.g., due to overload), release the locks immediately.
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
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.LAZY:
            return None
        return self._offload_kvcache_impl(index_meta)
        

    def finish_or_cancel_kvcache_ops(self) -> None:
        remain_tasks = list()
        for idx, task_handle in enumerate(self.ongoing_offload_tasks):
            wait_result = self.secondary_kvcache_manager.offload_wait_kvcache(task_handle)
            if wait_result.status in (
                SecondaryTaskStatus.LAUNCHED,
            ):
                remain_tasks.append(task_handle)
                print(f"[DEBUG] {wait_result.status}")
                continue

            if wait_result.status == SecondaryTaskStatus.READY:
                offload_success = self.secondary_kvcache_manager.finish_task(task_handle)
                print(f"[DEBUG] {wait_result.status}")
            elif wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            ):
                should_raise = False  # self.secondary_fail_policy == "fail_close"
                offload_success = False
                self.secondary_kvcache_manager.cancel_task(task_handle)
                if should_raise:
                    raise RuntimeError(f"{wait_result.status}")
                    pass
            else:
                raise RuntimeError(
                    f"Unexpected offload wait result status: {wait_result.status.value}, msg={wait_result.message}"
                )
            
            self.gpu_kvcache_mgr.release_offload_pages(
                *(self.secondary_kvcache_manager.get_offload_handle_metadata(task_handle)),
                offloaded=offload_success)
            
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
        # elif kvcache_config.secondary_backend == "flexkv":
        #     return FlexKVStorageManager(
        #         mode=flexkv_mode,
        #         server_addr=flexkv_server_addr,
        #         server_port=flexkv_server_port,
        #         num_layers=num_layers,
        #         num_heads=num_heads,
        #         head_dim=head_dim,
        #         page_size=page_size,
        #         secondary_wait_timeout_ms=secondary_wait_timeout_ms,
        #         secondary_fail_policy=secondary_fail_policy,
        #     )
        else:
            print(f"Unknown host kvcache backend {secondary_backend}")
        
        return NopSecondaryKVCacheManager()
        

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
