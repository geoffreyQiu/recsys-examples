import math
import os
import time

from kvcache_cpp import GPUKVCacheManagerImpl
import paged_kvcache_ops

import numpy as np
import torch
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .kvcache_config import KVCacheConfig
from .kvcache_utils import KVCacheOffloadMode, KVLookupResult, KVIndexMeta
from .kvcache_metadata import KVCacheMetadata, get_kvcache_metadata_buffer

class GPUKVCacheManager:
    _paged_kvcache_ops = None

    def __init__(self, 
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_tokens_per_page: int,
        num_tokens_per_chunk: int,
        num_primary_cache_pages: int,
        num_buffer_pages: int,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device_idx: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = num_tokens_per_page
        self.chunk_size = num_tokens_per_chunk
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_buffer_pages = num_buffer_pages
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.device_idx = device_idx
        self.kvcache_manager_impl = GPUKVCacheManagerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
            self.num_primary_cache_pages,
            self.num_buffer_pages,
            self.max_batch_size,
            self.max_sequence_length,
            self.device_idx,
        )
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
        # Auxiliary metadata
        self.num_sms = torch.cuda.get_device_properties(self.device_idx).multi_processor_count

    @staticmethod
    def _get_paged_kvcache_ops():
        if GPUKVCacheManager._paged_kvcache_ops is not None:
            return GPUKVCacheManager._paged_kvcache_ops
        try:
            import paged_kvcache_ops
        except ImportError as exc:
            raise ImportError(
                "Failed to import `paged_kvcache_ops`. "
                "If you see 'HostKVStorageImpl is already registered', "
                "it is a pybind type registration conflict between "
                "`kvcache_cpp` and `paged_kvcache_ops`."
            ) from exc
        GPUKVCacheManager._paged_kvcache_ops = paged_kvcache_ops
        return paged_kvcache_ops
    
    def lookup(self, uids: torch.Tensor) -> KVLookupResult:
        cached_start_indices_t, cached_lengths_t = self.kvcache_manager_impl.lookup(uids)
        return KVLookupResult(
            user_ids=uids,
            gpu_cached_start_indices=cached_start_indices_t.to(device=uids.device, dtype=torch.int32),
            gpu_cached_lengths=cached_lengths_t.to(device=uids.device, dtype=torch.int32),
        )

    def allocate(self, 
        uids: torch.Tensor, 
        seq_hist_lengths: torch.Tensor,  # total history lengths in the sequence of current batch
        lookup_results: KVLookupResult,
        output_kvcache_metadata: Optional[KVCacheMetadata] = None
    ) -> KVCacheMetadata:
        seq_hist_lengths_i32 = seq_hist_lengths.to(dtype=torch.int32)
        if lookup_results.cached_lengths is None:
            raise ValueError("lookup_results.cached_lengths is required for allocate()")
        cached_lengths_i32 = lookup_results.cached_lengths.to(
            device=seq_hist_lengths_i32.device, dtype=torch.int32
        )
        host_cached_lengths_i32 = (
            lookup_results.host_cached_lengths.to(
                device=seq_hist_lengths_i32.device, dtype=torch.int32
            )
            if lookup_results.host_cached_lengths is not None
            else torch.zeros_like(seq_hist_lengths_i32, dtype=torch.int32)
        )

        new_hist_lengths = seq_hist_lengths_i32 - cached_lengths_i32
        batch_size = int(uids.size(0))
        num_new_tokens = int(torch.sum(new_hist_lengths, dtype=torch.int32).item())

        if output_kvcache_metadata is None:
            print(f"[DEBUG] num_new_tokens: {num_new_tokens}")
            num_total_pages = torch.sum(
                torch.ceil(seq_hist_lengths_i32 / self.page_size), dtype=torch.int32
            ).item()
            output_kvcache_metadata = get_kvcache_metadata_buffer(
                batch_size,
                num_new_tokens,
                num_total_pages,
            )
        elif int(output_kvcache_metadata.kv_indptr.numel()) < batch_size + 1:
            raise ValueError(
                f"metadata.kv_indptr capacity {output_kvcache_metadata.kv_indptr.numel()} "
                f"is smaller than batch_size+1 {batch_size + 1}"
            )
        elif int(output_kvcache_metadata.batch_indices.numel()) < num_new_tokens:
            raise ValueError(
                f"metadata.batch_indices capacity {output_kvcache_metadata.batch_indices.numel()} "
                f"is smaller than num_new_tokens {num_new_tokens}"
            )

        # Keep Python-side metadata in sync with the current allocate call.
        output_kvcache_metadata.batch_size = batch_size
        output_kvcache_metadata.new_history_nnz = num_new_tokens
        output_kvcache_metadata.append_offsets = output_kvcache_metadata.new_history_offsets
        output_kvcache_metadata.max_seqlen = int(torch.max(seq_hist_lengths_i32).item())
        self.kvcache_manager_impl.allocate(
            uids, 
            seq_hist_lengths_i32,
            host_cached_lengths_i32,
            output_kvcache_metadata.page_ids_gpu_buffer,
            output_kvcache_metadata.metadata_gpu_buffer,
        )
        return output_kvcache_metadata        
    
    def evict(self, uids: torch.Tensor) -> None:
        for uid in uids.tolist():
            self.kvcache_manager_impl.evict(uid)
    
    def evict_all(self) -> None:
        self.kvcache_manager_impl.evict_all()
    
    # def invalid(self, uids) -> None:
    #     for uid in uids.tolist():
    #         self.kvcache_manager_impl.invalid(uid)
    
    def put(self, k, v, layer_idx, kvcache_metadata: KVCacheMetadata, append_offsets: Optional[torch.Tensor] = None) -> None:
        assert k.shape == v.shape, f"key and value shape mismatch: {k.shape} vs {v.shape}"
        if k.size(0) == self.num_layers:
            raise NotImplementedError("Only support layer-wise in this implementation.")
        if k.dim() != 3:
            raise ValueError(f"Expected 3D k/v tensor [N, H, D], got {k.shape}")
        if int(k.size(1)) != self.num_heads or int(k.size(2)) != self.head_dim:
            raise ValueError(
                f"Input k/v shape {k.shape} incompatible with heads/head_dim "
                f"{self.num_heads}/{self.head_dim}"
            )

        append_offsets = (
            kvcache_metadata.append_offsets
            if kvcache_metadata.append_offsets is not None
            else kvcache_metadata.new_history_offsets
        )
        if append_offsets is None:
            raise ValueError("KVCacheMetadata.append_offsets/new_history_offsets is required for put()")
        batch_size = int(
            kvcache_metadata.batch_size
            if kvcache_metadata.batch_size > 0
            else int(kvcache_metadata.kv_indptr.numel()) - 1
        )
        if int(append_offsets.numel()) < batch_size + 1:
            raise ValueError(
                f"append_offsets length {append_offsets.numel()} < batch_size+1 {batch_size + 1}"
            )

        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_table[layer_idx].unbind(dim=1)
        paged_kvcache_ops = self._get_paged_kvcache_ops()
        assert k.shape[-2:] == paged_k_cache.shape[-2:], f"input k/v shape {k.shape} mismatch with cache shape {paged_k_cache.shape}"
        paged_kvcache_ops.append_kvcache(
            k, v, 
            kvcache_metadata.batch_indices,
            kvcache_metadata.position,
            append_offsets[: batch_size + 1],
            kvcache_metadata.new_history_nnz_cuda,
            kvcache_metadata.new_history_nnz,
            paged_k_cache,
            paged_v_cache,
            kvcache_metadata.kv_indices,
            kvcache_metadata.kv_indptr,
            kvcache_metadata.kv_last_page_len,
            0,  # NHD layout
            self.num_sms)
    
    def get(self, page_ids, last_page_lens, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: debug interface (for symmetry). Not used in inference pipeline.
        (paged_k_cache, paged_v_cache) = self.gpu_kvcache_table[layer_idx].unbind(dim=1)

        k = paged_k_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        v = paged_v_cache[page_ids].view(-1, self.num_heads, self.head_dim).clone()
        k = k[ : k.size(0) - (self.page_size - int(last_page_lens)), ... ]
        v = v[ : v.size(0) - (self.page_size - int(last_page_lens)), ... ]
        return k, v

    def revoke_onboard_pages(self, user_ids, onboard_page_starts, num_onboard_pages):
        self.kvcache_mananger_impl.revoke_onboard_pages(user_ids, onboard_page_starts, num_onboard_pages)
    
    # [[ offload critirea ]]
    def check_for_offload(self, uids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.kvcache_manager_impl.check_for_offload(
            uids if uids is not None else torch.tensor([], dtype=torch.int64))
    
    def acquire_offload_pages(self, uids: torch.Tensor, offloaded_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        return self.kvcache_manager_impl.acquire_offload_pages(uids, offloaded_lengths)
    
    def release_offload_pages(self, 
        uids: torch.Tensor, 
        offload_start_indices: torch.Tensor, 
        offload_lengths: torch.Tensor,
        offloaded: bool,
    ) -> None:
        return self.kvcache_manager_impl.release_offload_pages(
            uids, offload_start_indices, offload_lengths, offloaded)