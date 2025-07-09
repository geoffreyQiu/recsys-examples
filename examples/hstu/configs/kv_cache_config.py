# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import List, Optional

import torch

from . import HSTUConfig


@dataclass
class KVCacheMetadata:
    """
    KVCacheMetadata is a  data class for the HSTU KV cache metadata of a batch.
    """

    # paged cache metadata
    kv_indices: torch.Tensor = None
    kv_indptr: torch.Tensor = None
    kv_last_page_len: torch.Tensor = None
    total_history_lengths: torch.Tensor = None
    total_history_offsets: torch.Tensor = None

    # appending metadata
    batch_indices: torch.Tensor = None
    position: torch.Tensor = None
    new_history_nnz: int = None
    new_history_nnz_cuda: torch.Tensor = None

    # onload utility
    onload_history_kv_buffer: List[torch.Tensor] = None
    onload_history_kv_events: List[torch.cuda.Event] = None

    # paged cache table pointers
    kv_cache_table: List[torch.Tensor] = None


@dataclass
class KVCacheConfig:
    """
    KVCacheConfig is a configuration data class for the HSTU KV cache.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        page_size (int): The number of tokens per cache page.
        offload_chunksize (int): The size of basic offload data chunk.
        max_batch_size (int): The maximum batch size for the inference input.
        max_seq_len (int): The upper bound of sequence length for each sequence in the inference batch.
        max_attention_window (int): (Optional) The maximum window size for HSTU attention calculation.
    """

    blocks_in_primary_pool: int
    page_size: int
    offload_chunksize: int
    max_batch_size: int
    max_seq_len: int
    max_attention_window: Optional[int] = None


def get_kvcache_config(
    blocks_in_primary_pool: int,
    page_size: int,
    offload_chunksize: int,
    max_batch_size: int,
    max_seq_len: int,
    max_attention_window: Optional[int] = None,
) -> KVCacheConfig:
    """
    Create the HSTU KV cache configuration.

    Args:
        blocks_in_primary_pool (int): The number of cache pages per layer.
        page_size (int): The number of tokens per cache page in the paged KV cache.
        offload_chunksize (int): The size of basic offload data chunk.
        max_batch_size (int): The max batch size.
        max_gpu_cache_seqlen (int): The upper bound of sequence length in gpu cache.
        max_host_cache_seqlen (int): The upper bound of sequence length in host cache.
        max_attention_window (int): The max attention window size.

    Returns:
        KVCacheConfig: The HSTU KV cache configuration object.
    """

    return KVCacheConfig(  # type: ignore
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_attention_window=max_attention_window,
    )


def get_kvcache_metadata_buffer(hstu_config: HSTUConfig, kvcache_config: KVCacheConfig):
    device = torch.cuda.current_device()
    torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32

    max_new_history_seqlen = kvcache_config.max_batch_size * kvcache_config.max_seq_len
    max_num_pages_per_seq = (
        kvcache_config.max_seq_len
        + kvcache_config.max_seq_len
        + kvcache_config.page_size
        - 1
    ) // kvcache_config.page_size
    max_host_kv_buffer_size = (
        kvcache_config.max_batch_size * kvcache_config.max_seq_len,
        hstu_config.num_attention_heads * hstu_config.kv_channels,
    )

    default_num_pages_per_seq = 4
    paged_indices_buffer = torch.randint(
        kvcache_config.blocks_in_primary_pool,
        (kvcache_config.max_batch_size * max_num_pages_per_seq,),
        dtype=torch.int32,
        device=device,
    )
    page_indptr_buffer = (
        torch.arange(
            kvcache_config.max_batch_size + 1, dtype=torch.int32, device=device
        )
        * default_num_pages_per_seq
    )
    last_page_lens_buffer = torch.full(
        (kvcache_config.max_batch_size,),
        kvcache_config.page_size,
        dtype=torch.int32,
        device=device,
    )
    batch_indices_buffer = torch.zeros(
        (max_new_history_seqlen,), dtype=torch.int32, device=device
    )
    position_buffer = torch.zeros(
        (max_new_history_seqlen,), dtype=torch.int32, device=device
    )
    total_history_offsets_buffer = (
        torch.arange(
            kvcache_config.max_batch_size + 1, dtype=torch.int32, device=device
        )
        * default_num_pages_per_seq
        * kvcache_config.page_size
    )
    return KVCacheMetadata(
        kv_indices=paged_indices_buffer,
        kv_indptr=page_indptr_buffer,
        kv_last_page_len=last_page_lens_buffer,
        batch_indices=batch_indices_buffer,
        position=position_buffer,
        new_history_nnz=max_new_history_seqlen,
        new_history_nnz_cuda=torch.ones((1,), dtype=torch.int32, device=device),
        total_history_offsets=total_history_offsets_buffer,
        onload_history_kv_buffer=[],
        onload_history_kv_events=[],
    )


def copy_kvcache_metadata(dst_metadata: KVCacheMetadata, src_metata: KVCacheMetadata):
    def copy_tensor(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0

    def copy_offsets(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]

    copy_tensor(dst_metadata.kv_indices, src_metata.kv_indices)
    copy_offsets(dst_metadata.kv_indptr, src_metata.kv_indptr)
    copy_tensor(dst_metadata.kv_last_page_len, src_metata.kv_last_page_len)
    copy_tensor(dst_metadata.batch_indices, src_metata.batch_indices)
    copy_tensor(dst_metadata.position, src_metata.position)
    copy_tensor(dst_metadata.new_history_nnz_cuda, src_metata.new_history_nnz_cuda)
    copy_offsets(dst_metadata.total_history_offsets, src_metata.total_history_offsets)

    dst_metadata.new_history_nnz = src_metata.new_history_nnz
