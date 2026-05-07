import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from configs import KVCacheMetadata
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class KVCacheManager:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        kv_headdim,
        num_tokens_per_page,
        num_primary_cache_pages,
        num_onload_buffer_pages,
        num_reserved_buffer_pages,
        num_tokens_per_chunk,
        max_num_sequences,
        max_sequence_length,
        max_batch_size,
        max_queued_offload_tokens,
        num_onload_buffer_chunks=1,
        num_offload_buffer_chunks=8,
        num_memcpy_workers=8,
        enable_nvcomp=False,
        secondary_kvcache_manager: Optional[SecondaryKVCacheManagerBase] = None,
        offload_mode: str = "lazy",
        secondary_wait_timeout_ms: int = 0,
        secondary_fail_policy: str = "fail_open",

    ):
        self.num_layers = num_layers
        self.num_heads = num_kv_heads
        self.head_dim = kv_headdim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages
        self.max_batch_size = max_batch_size

        self.gpu_kvcache_mgr = GPUKVCacheManager(
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=kv_headdim,
            num_tokens_per_page=num_tokens_per_page,
            num_tokens_per_chunk=num_tokens_per_chunk,
            num_primary_cache_pages=num_primary_cache_pages,
            num_buffer_pages=num_buffer_pages,
            max_batch_size=max_batch_size,
            dtype=torch.bfloat16,
            device_idx=torch.cuda.current_device(),
        )
        self.dummy_empty_tensor = torch.tensor([], dtype=torch.int32)

        self.secondary_kvcache_manager = secondary_kvcache_manager if secondary_kvcache_manager is not None else NopSecondaryKVCacheManager()
        self.secondary_kvcache_manager.register_gpu_cache_tensors(self.gpu_kvcache_mgr.gpu_kvcache_table)

        self.offload_mode = (
            KVCacheOffloadMode(offload_mode)
            if offload_mode in {m.value for m in KVCacheOffloadMode}
            else KVCacheOffloadMode.LAZY
        )
        self.secondary_wait_timeout_ms = int(secondary_wait_timeout_ms)
        self.secondary_fail_policy = secondary_fail_policy
        self.ongoing_onboard_tasks: Dict[str, SecondaryTaskHandle] = {}
        self.ongoing_offload_tasks: Dict[str, SecondaryTaskHandle] = {}

    @staticmethod
    def _build_secondary_manager_from_config(
        secondary_backend: str,
        flexkv_mode: str,
        flexkv_server_addr: str,
        flexkv_server_port: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        secondary_wait_timeout_ms: int,
        secondary_fail_policy: str,
    ) -> SecondaryKVCacheManagerBase:
        if secondary_backend == "flexkv":
            return FlexKVStorageManager(
                mode=flexkv_mode,
                server_addr=flexkv_server_addr,
                server_port=flexkv_server_port,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                page_size=page_size,
                secondary_wait_timeout_ms=secondary_wait_timeout_ms,
                secondary_fail_policy=secondary_fail_policy,
            )
        return NopSecondaryKVCacheManager()

    
    def lookup_kvcache(
        self,
        user_ids: Union[List[int], torch.Tensor],
        history_sequence_lengths: Union[List[int], torch.Tensor],
    ) -> KVLookupResult:
        _user_ids, _total_history_lengths = self._normalize_user_ids_and_lengths(
            user_ids, history_sequence_lengths
        )
        gpu_lookup_results = self.gpu_kvcache_mgr.lookup(user_ids)
        index_meta = self.secondary_kvcache_manager.build_index_meta(user_ids, history_sequence_lengths)
        host_lookup_results = self.secondary_kvcache_manager.lookup_kvcache(index_meta)

        lookup_results = KVLookupResult.merge(gpu_lookup_results, host_lookup_results)
        return lookup_results

    def allocate_kvcache(
        self,
        lookup_results: KVLookupResult,
    ) -> KVCacheMetadata:
        return self.gpu_kvcache_mgr.allocate(lookup_results)
    
    def onboard_launch_kvcache(
        self,
        kv_index_meta: KVIndexMeta,
    ) -> SecondaryTaskHandle:
        task_handle = self.secondary_kvcache_manager.onboard_launch_kvcache(
            kv_index_meta, kv_index_meta.restore_slot_mapping
        )
        rid = kv_index_meta.request_id
        self.ongoing_onboard_tasks[rid] = task_handle
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
            self.secondary_kvcache_manager.cancel_task(task_handle)
            self.ongoing_onboard_tasks.pop(kv_index_meta.request_id, None)
            if self.secondary_fail_policy == "fail_close":
                raise RuntimeError(
                    f"onboard wait failed: status={wait_result.status.value}, msg={wait_result.message}"
                )
        elif wait_result.ready:
            self.ongoing_onboard_tasks.pop(kv_index_meta.request_id, None)
        return wait_result

    
    def _offload_kvcache_impl(
        self,
        index_meta: KVIndexMeta,
    ):
        # 0. Maybe r
        # 1. Get the maximum batch for offloading from GPU
        uids_to_offload = self.gpu_kvcache_mgr.check_for_offload(index_meta.user_ids)

        # 2. Lookup host again in case of multi-GPU instances
        offloaded_lengths = self.secondary_kvcache_manager.lookup_kvcache(index_meta)

        # 3. Acquire and lock GPU cache pages
        (offload_user_ids, 
         offload_start_indices, 
         offload_page_indices_list, 
        ) = self.gpu_kvcache_mgr.acquire_offload_pages(uids_to_offload, offloaded_lengths)
        # returned with cache pages locked (per-uid). TODO(junyiq): per-(uid, lock_start, lock_end)
        # Note: all pages from offload_page_indicesa are required to offload. duplication will be resolved when finished.
        if offload_user_ids.size(0) == 0:
            return None

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
        
        rid = str(uuid4())
        self.ongoing_offload_tasks[rid] = task_handle
        return task_handle
    
    def eager_offboard_kvcache(
        self,
        user_ids: torch.Tensor,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.EAGER:
            return None
        return self._offload_kvcache_impl(user_ids)


    def lazy_offload_kvcache(
        self,
        user_ids: torch.Tensor,
    ) -> Optional[SecondaryTaskHandle]:
        if self.offload_mode != KVCacheOffloadMode.LAZY:
            return None
        return self._offload_kvcache_impl(user_ids)
        

    def finish_or_cancel_kvcache_ops(self, kv_index_meta=None) -> None:
        request_ids = list(self.ongoing_offload_tasks.keys())
        for request_id in request_ids:
            task_handle = self.ongoing_offload_tasks.get(request_id)
            wait_result = self.secondary_kvcache_manager.offload_wait_kvcache(task_handle)

            if wait_result.status == SecondaryTaskStatus.READY:
                offload_success = self.secondary_kvcache_manager.finish_task(task_handle)
                
            elif wait_result.status in (
                SecondaryTaskStatus.FAILED,
                SecondaryTaskStatus.TIMEOUT,
                SecondaryTaskStatus.CANCELLED,
            ):
                # should_raise = failed and self.secondary_fail_policy == "fail_close"
                should_raise = False
                offload_success = False
                self.secondary_kvcache_manager.cancel_task(task_handle)
                if should_raise:
                    pass
            else:
                raise RuntimeError(
                    f"Unexpected offload wait result status: {wait_result.status.value}, msg={wait_result.message}"
                )
            
            self.ongoing_offload_tasks.pop(request_id, None)
            self.gpu_kvcache_mgr.release_offload_pages(
                    task_handle.get_user_ids(), task_handle.get_start_indices(), task_handle.get_lengths(),
                    offloaded=offload_success)


    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)

        num_cached = torch.clamp_min(origin_num_cached - num_context, 0).to(torch.int32)
        num_cached_action = num_cached // 2
        num_cached_item = num_cached - num_cached_action
        num_hist_cached = torch.concat([num_cached_item, num_cached_action], dim=0)

        old_offsets = batch.features.offsets().cpu()
        old_lengths = batch.features.lengths().cpu()

        item_offset = num_context * batch.batch_size

        new_lengths = torch.zeros_like(old_lengths)
        new_lengths[:item_offset] = torch.where(
            (origin_num_cached == 0).view(-1, batch.batch_size),
            old_lengths[:item_offset].view(-1, batch.batch_size),
            new_lengths[:item_offset].view(-1, batch.batch_size),
        ).view(-1)
        new_lengths[item_offset:] = old_lengths[item_offset:] - num_hist_cached

        startpos = (
            old_offsets[item_offset : item_offset + 2 * batch.batch_size]
            + num_hist_cached
        )
        endpos = old_offsets[item_offset + 1 :]

        old_values = batch.features.values()
        new_hist_value = [
            old_values[startpos[idx] : endpos[idx]]
            for idx in range(2 * batch.batch_size)
        ]

        new_context_value = [
            old_values[idx : idx + 1]
            for idx in range(num_context * batch.batch_size)
            if int(new_lengths[idx]) > 0
        ]

        new_features = KeyedJaggedTensor(
            values=torch.cat(new_context_value + new_hist_value, dim=0),
            lengths=new_lengths.cuda(),
            keys=batch.features.keys(),
        )
        batch.features = new_features

        torch.cuda.nvtx.range_pop()
        return batch

    @classmethod
    def from_config(cls, hstu_config, kvcache_config):
        if kvcache_config.max_queued_offload_tokens is None:
            kvcache_config.max_queued_offload_tokens = (
                4 * hstu_config.max_batch_size * hstu_config.max_seq_len
            )

        secondary_mgr = cls._build_secondary_manager_from_config(
            secondary_backend=getattr(kvcache_config, "secondary_backend", "nop"),
            flexkv_mode=getattr(kvcache_config, "flexkv_mode", "direct"),
            flexkv_server_addr=getattr(kvcache_config, "flexkv_server_addr", ""),
            flexkv_server_port=getattr(kvcache_config, "flexkv_server_port", 0),
            num_layers=hstu_config.num_layers,
            num_heads=hstu_config.num_heads,
            head_dim=hstu_config.head_dim,
            page_size=kvcache_config.page_size,
            secondary_wait_timeout_ms=getattr(kvcache_config, "secondary_wait_timeout_ms", 0),
            secondary_fail_policy=getattr(kvcache_config, "secondary_fail_policy", "fail_open"),
        )
        return cls(
            hstu_config.num_layers,
            hstu_config.num_heads,
            hstu_config.head_dim,
            kvcache_config.page_size,
            kvcache_config.blocks_in_primary_pool,
            math.ceil(
                hstu_config.max_batch_size
                * hstu_config.max_seq_len
                / kvcache_config.page_size
            ),
            0,
            kvcache_config.offload_chunksize,
            -1,
            hstu_config.max_seq_len,
            hstu_config.max_batch_size,
            kvcache_config.max_queued_offload_tokens,
            kvcache_config.num_onload_buffer_chunks,
            kvcache_config.num_offload_buffer_chunks,
            kvcache_config.num_memcpy_workers,
            kvcache_config.enable_nvcomp,
            secondary_mgr,
            getattr(kvcache_config, "offload_mode", "lazy"),
            getattr(kvcache_config, "secondary_wait_timeout_ms", 0),
            getattr(kvcache_config, "secondary_fail_policy", "fail_open"),
        )
