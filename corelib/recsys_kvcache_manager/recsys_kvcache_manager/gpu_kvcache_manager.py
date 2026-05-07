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


class GPUKVCacheManager:
    # Refer: https://github.com/NVIDIA/recsys-examples/blob/main/examples/commons/ops/cuda_ops/csrc/paged_kvcache_ops_cuda.cpp#L285C30-L306C42
    def __init__(self, 
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        num_primary_cache_pages: int,
        num_buffer_pages: int,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device_idx: int = -1):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device_idx = device_idx

        self.gpu_kvcache_table: torch.Tensor = torch.empty(
            [
                self.num_layers,
                self.num_primary_cache_pages,
                2,  # k/v
                self.page_size,
                self.num_heads,
                self.head_dim,
            ],
            dtype=self.dtype,
            device=self.device_idx,
        )
        self.kvcache_mananger_impl = GPUKVCacheManagerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.num_primary_cache_pages,
            self.num_buffer_pages,
            self.max_batch_size,
            self.device_idx,
        )

        # Auxiliary metadata
        self.num_sms = torch.cuda.get_device_properties(self.device_idx).multi_processor_count
    
    def lookup(self, uids: torch.Tensor) -> "LookupResults":
        cached_start_indices, cached_lengths = self.kvcache_mananger_impl.lookup(uid)
        return KVLookupResult(
            request_id="",
            user_ids=uids,
            gpu_cached_start_indices=cached_start_indices,
            gpu_cached_lengths=cached_lengths,
        )
    
    def allocate(self, 
        uids: torch.Tensor, 
        seq_history_lengths: torch.Tensor, 
        lookup_results: KVLookupResult,
        output_kvcache_metadata: KVCacheMetadata) -> KVCacheMetadata:
        if output_kvcache_metadata is None:
            batch_size = uids.size(0)
            num_total_pages = torch.sum(torch.ceil(seq_history_lengths / self.page_size), dtype=torch.int32).item()
            output_kvcache_metadata = KVCacheMetadata(
                page_ids_gpu_buffer=torch.empty((num_total_pages,), dtype=torch.int32, device=self.device_idx),
                metadata_gpu_buffer=torch.empty((batch_size * 5 + 4,), dtype=torch.int32, device=self.device_idx),
            )
        self.kvcache_mananger_impl.allocate(
            uids, 
            seq_history_lengths,
            lookup_results.host_cached_lengths,
            output_kvcache_metadata.page_ids_gpu_buffer,
            output_kvcache_metadata.metadata_gpu_buffer)
        return output_kvcache_metadata
    
    def evict(self, uids) -> None:
        for uid in uids.tolist():
            self.kvcache_mananger_impl.evict(uid)
    
    def invalid(self, uids) -> None:
        for uid in uids.tolist():
            self.kvcache_mananger_impl.invalid(uid)
    
    def put(self, k, v, layer_idx, kvcache_metadata: KVCacheMetadata) -> None:
        assert k.shape == v.shape, f"key and value shape mismatch: {k.shape} vs {v.shape}"
        if k.size(0) == self.num_layers:
            raise NotImplementedError("Only support layer-wise in this implementation.")
        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_table[layer_idx].unbind(dim=1)
        assert k.shape == paged_k_cache.shape, f"input k/v shape {k.shape} mismatch with cache shape {paged_k_cache.shape}"
        paged_kvcache_ops.append_kvcache(
            k, v, 
            kvcache_metadata.batch_indices,
            kvcache_metadata.position,
            kvcache_metadata.append_offsets[: kvcache_metadata.batch_size + 1],
            kvcache_metadata.new_history_nnz_cuda,
            kvcache_metadata.new_history_nnz,
            paged_k_cache,
            paged_v_cache,
            kvcache_metadata.kv_indices,
            kvcache_metadata.kv_indptr,
            kvcache_metadata.kv_last_page_len,
            0,  # NHD layout
            self.num_sms)
    
    def get(self, kvcache_metadata, user_ids, user_indices, layer_indices) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: debug interface (for symmetry). Not used in inference pipeline.
        pass
    
    # [[ offload critirea ]]
    def check_for_offload(self, uids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.kvcache_mananger_impl.check_for_offload(
            uids if uids is not None else torch.tensor([], dtype=torch.int64))
    
    def acquire_offload_pages(self, uids: torch.Tensor, offloaded_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        return self.kvcache_mananger_impl.acquire_offload_pages(uids, offloaded_lengths)
    
    def release_offload_pages(self, 
        uids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_lengths: torch.Tensor,
        offloaded: bool,
    ) -> None:
        return self.kvcache_mananger_impl.release_offload_pages(
            uids, offload_start_indices, offload_lengths, offloaded)