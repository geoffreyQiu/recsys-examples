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


class NativeHostKVCacheManager(SecondaryKVCacheManagerBase):
    def __init__(self):
        self._offload_timeout = 1.0 # seconds
        self.kvcache_mananger_impl = NativeHostKVCacheManagerImpl(...)
    
    def build_index_meta(self, user_ids: torch.Tensor, history_sequence_lengths: torch.Tensor) -> KVIndexMeta:
        index_meta = KVIndexMeta(
            user_ids=user_ids,
        )
        return index_meta

    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        cached_start_indices, cached_lengths = self.kvcache_mananger_impl.get_kvdata_length(index_meta.user_ids)
        return KVLookupResult(
            request_id="",
            user_ids=index_meta.user_ids,
            host_cached_start_indices=cached_start_indices,
            host_cached_lengths=cached_lengths,
        )

    def onboard_launch_kvcache(self,
        index_meta: KVIndexMeta,
        lookup_result: KVLookupResult,
        kvcache_metadata: KVCacheMetadata
    ) -> SecondaryTaskHandle:
        native_handle = NativeOnloadHandle(self.num_layers)
        onload_lengths = torch.where(
            lookup_result.gpu_cached_lengths==0, lookup_result.host_cached_lengths, lookup_result.gpu_cached_start_indices
        )
        onload_paged_ids_list = [
            kvcache_metadata.page_ids.narrow(
                '''dim=''' 0,
                '''start=''' kvcache_metadata.page_indptrs[seq_idx], # gpu, this is allowed to be Tensor. TODO(junyiq): check if there is d2h
                '''length=''' onload_lengths[seq_idx].item(),
            ) for seq_idx in range(index_meta.user_ids.size(0))      
        ]
        if torch.sum(onload_lengths).item() == 0:
            # No data to onboard, skip the task.
            return SecondaryTaskHandle(
                backend="native",
                handle=native_handle,
                status=SecondaryTaskStatus.SKIPPED,
                metadata={"request_id": index_meta.request_id},
            )
        self.kvcache_mananger_impl.onload_kvcache(
            index_meta.user_ids,
            onload_paged_ids_list,
            native_handle)
        return SecondaryTaskHandle(
            backend="native",
            handle=native_handle,
            status=SecondaryTaskStatus.LAUNCHED,
            metadata={"request_id": index_meta.request_id},
        )

    def onboard_wait_kvcache(self, task_handle):
        # Use layerwise data transfer and sync for native backend.
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.SKIPPED,
            ready=False,
        )
    
    def onboard_wait_kvcache_by_layer(self, task_handle, layer_idx: int):  # wait for a single layer
        task_handle.handle.wait_host(layer_idx)
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.EVENT_READY,  # the following get on the default stream will be synced
            ready=True,
        )

    def offload_launch_kvcache(
        self,
        offload_user_ids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_page_indices_list: List[torch.Tensor], 
    ):
        native_handle = NativeOffloadHandle(self.num_layers)
        ret = self.kvcache_mananger_impl.offload_kvcache(
            offload_user_ids, 
            offload_start_indices, 
            offload_page_indices_list, 
            native_handle)
        if not ret:
            return SecondaryTaskHandle(
                backend="native",
                handle=native_handle,
                status=SecondaryTaskStatus.FAILED,
                time_launched=time.time()
        )
        return SecondaryTaskHandle(
            backend="native",
            handle=native_handle,
            status=SecondaryTaskStatus.LAUNCHED,
            time_launched=time.time()
        )

    def offload_wait_kvcache(self, task_handle):
        is_ready = task_handle.handle.is_ready()
        elapsed_time = time.time() - task_handle.time_launched
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY if is_ready else 
                   SecondaryTaskStatus.TIMEOUT if elapsed_time > self._offload_timeout else 
                   SecondaryTaskStatus.WAITING,
            ready=is_ready,
        )
    
    def finish_task(self, task_handle):
        if isinstance(task_handle.handle, NativeOnloadHandle):
            raise NotImplementedError("Finish onload by layer is supported, but not the whole task at once, since the native implementation uses layerwise synchronization.")
        elif isinstance(task_handle.handle, NativeOffloadHandle):
            return self.kvcache_mananger_impl.finish_offload(task_handle.user_ids, task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")
        return False

    def cancel_task(self, task_handle):
        if isinstance(task_handle.handle, NativeOnloadHandle):
            raise NotImplementedError("Cancel onload by layer is not supported in the current native implementation.")
        elif isinstance(task_handle.handle, NativeOffloadHandle):
            return self.kvcache_mananger_impl.cancel_offload(task_handle.handle)
        else:
            raise ValueError(f"Unknown task handle type: {type(task_handle.handle)}")
        return False
    
    def register_gpu_cache_tensors(self, cache_table_list: Union[torch.Tensor, List[torch.Tensor]]):
        # If the native backend also manages GPU KV cache, it can register the cache tensors here.
        # For example, it can pass the tensor pointers to the native library for direct GPU access.
        assert isinstance(cache_table_list, list), "Currently only support per-layer GPU cache tensor for native backend."
        self.kvcache_mananger_impl.set_gpu_kvcache_table(cache_table_list)