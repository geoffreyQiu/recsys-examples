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
        num_primary_cache_pages: int,
        num_onload_buffer_pages: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_onload_buffer_pages = num_onload_buffer_pages
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cuda")

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
            device=self.device,
        )
        self.kvcache_mananger_impl = GPUKVCacheManagerImpl(
            self.num_layers,
            self.page_size,
            self.num_primary_cache_pages,
            self.num_onload_buffer_pages,
        )

        # Auxiliary metadata
        self.num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
    
    def lookup(self, uids: torch.Tensor) -> "LookupResults":
        cached_start_indices, cached_lengths = self.kvcache_mananger_impl.lookup(uid)
        return KVLookupResult(
            request_id="",
            user_ids=uids,
            gpu_cached_start_indices=cached_start_indices,
            gpu_cached_lengths=cached_lengths,
        )
    
    def allocate(self, uids: torch.Tensor, seq_history_lengths: torch.Tensor, output_kvcache_metadata: KVCacheMetadata) -> KVCacheMetadata:
        if output_kvcache_metadata is None:
            batch_size = uids.size(0)
            num_total_pages = torch.sum(torch.ceil(seq_history_lengths / self.page_size), dtype=torch.int32).item()
            output_kvcache_metadata = KVCacheMetadata(
                page_ids_gpu_buffer=torch.empty((num_total_pages,), dtype=torch.int32, device=self.device),
                metadata_gpu_buffer=torch.empty((batch_size * 5 + 4,), dtype=torch.int32, device=self.device),
            )
        self.kvcache_mananger_impl.allocate(
            uids, 
            seq_history_lengths,
            output_kvcache_metadata.page_ids_gpu_buffer,
            output_kvcache_metadata.metadata_gpu_buffer)
        return output_kvcache_metadata
    
    def evict(self, uids, kvcache_metadata: KVCacheMetadata) -> None:
        self.kvcache_mananger_impl.evict(uids)
    
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
    def check_offload(self, uids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # return offload_user_ids
        
        # 1. if a new offload chunk is filled, offload [eager]
        # 2. if ( num_empty_pages <= 4 * num_pages_per_batch ), offload [lazy]
        offload_user_ids = self.kvcache_mananger_impl.check_offload(uids)
        return offload_user_ids