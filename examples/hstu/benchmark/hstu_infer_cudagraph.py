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
import commons.utils.initialize as init
import math
import pytest
import torch
from configs import (
    RankingConfig,
    ShardedEmbeddingConfig,
    get_hstu_config,
    get_kvcache_config,
    KVCacheMetadata,
    copy_kvcache_metadata,
)
from dataset.utils import FeatureConfig, RankingBatch, RetrievalBatch
from megatron.core import tensor_parallel
from model.ranking_gr_infer import RankingGRInferenceModel, copy_jagged_metadata
from modules.jagged_data import JaggedData
import torch.nn.functional as F
from hstu_attn import hstu_attn_varlen_func
from einops import rearrange, repeat
from typing import Optional, Tuple
import append_kvcache
import flashinfer

def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i]))
    indices = torch.cat(indices)
    output = torch.zeros((batch * seqlen), *unpadded_input.shape[1:], device=unpadded_input.device, dtype=unpadded_input.dtype)
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def pad_input_delta_q(unpadded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    indices = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        indices.append(torch.arange(seqlen * i + act_seqlen_k - act_seqlen_q, seqlen * i + act_seqlen_k))
    indices = torch.cat(indices)
    output = torch.zeros((batch * seqlen), *unpadded_input.shape[1:], device=unpadded_input.device, dtype=unpadded_input.dtype)
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, :(cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)

def unpad_input_delta_q(padded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        output.append(padded_input[i, act_seqlen_k - act_seqlen_q:act_seqlen_k, :])
    return torch.cat(output, dim=0)

def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    is_delta_q: bool = False,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    dtype_out = q.dtype
    if is_delta_q:
        padded_q = pad_input_delta_q(q, q_offsets, k_offsets, B, seqlen_k)
    else:
        padded_q = pad_input(q, q_offsets, B, seqlen_q)
    padded_k = pad_input(k, k_offsets, B, seqlen_k)
    padded_v = pad_input(v, k_offsets, B, seqlen_k)


    padded_q = padded_q.view(B, seqlen_q, num_heads, attention_dim)
    padded_k = padded_k.view(B, seqlen_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, seqlen_k, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
        if rab is not None:
            rab = rab.float()
    qk_attn = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k,)

    if rab is not None:
        padding = (0, qk_attn.shape[-1]-rab.shape[-1], 0, qk_attn.shape[-2]-rab.shape[-2])
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / seqlen_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        # masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :]
        ext_invalid_attn_mask=torch.zeros_like(masked_qk_attn)
        d0,d1,d2,d3=invalid_attn_mask.shape
        d1 = masked_qk_attn.shape[1]
        ext_invalid_attn_mask[:d0,:d1,:d2,:d3].copy_(invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :])
        masked_qk_attn = masked_qk_attn * ext_invalid_attn_mask[:,:,:,:]

    attn_output = torch.einsum("bhnm,bmhd->bnhd", masked_qk_attn, padded_v,)

    attn_output = attn_output.reshape(B, seqlen_q, num_heads * linear_dim)
    if is_delta_q:
        attn_output = unpad_input_delta_q(attn_output, q_offsets, k_offsets, B, seqlen_k)
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output.to(dtype_out)

def _hstu_paged_kv_attention(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    kv_cache: torch.Tensor = None,
    page_offsets: torch.Tensor = None,
    page_ids: torch.Tensor = None,
    last_page_lens: torch.Tensor = None,
):
    k_con = torch.empty((0, num_heads, attention_dim), device=k.device, dtype=k.dtype)
    v_con = torch.empty((0, num_heads, attention_dim), device=v.device, dtype=v.dtype)

    for i in range(len(last_page_lens)):
        page_num = page_offsets[i + 1] - page_offsets[i]
        new_history_len = q_offsets[i + 1] - q_offsets[i] - num_targets[i]
        for j in range(page_num - 1):
            k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i] + j], 0, :, :, :]), dim=0)
            v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i] + j], 1, :, :, :]), dim=0)
        k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 0, :last_page_lens[i], :, :]), dim=0)
        k_con = torch.cat((k_con, k[(q_offsets[i] + new_history_len):q_offsets[i + 1], :, :]), dim=0)
        v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 1, :last_page_lens[i], :, :]), dim=0)
        v_con = torch.cat((v_con, v[(q_offsets[i] + new_history_len):q_offsets[i + 1], :, :]), dim=0)

    return _hstu_attention_maybe_from_cache(
              num_heads=num_heads,
              attention_dim=attention_dim,
              linear_dim=linear_dim,
              seqlen_q=seqlen_q,
              seqlen_k=seqlen_k,
              q=q,
              k=k_con,
              v=v_con,
              q_offsets=q_offsets,
              k_offsets=k_offsets,
              rab=None,
              invalid_attn_mask=invalid_attn_mask,
              alpha=alpha,
              upcast=upcast,
              is_delta_q=False,
          )

def get_offsets_from_lengths(lengths):
    offsets = torch.zeros((lengths.shape[0]+1,), dtype=lengths.dtype, device=lengths.device)
    torch.cumsum(lengths, 0, out=offsets[1:])
    return offsets

def generate_kvdata_testcase(
    max_seq_len: int,
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int):

    lengths = torch.randint(max_seq_len//2, max_seq_len+1, (batch_size,), dtype=torch.int32)
    num_tokens = torch.sum(lengths).item()
    values = torch.randn((num_layers, 2, num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=torch.cuda.current_device())
    print("generated kvdata", values.shape, lengths)
    return (values, lengths)

def setup_kvcache_testcase(
    gpu_kvcache_manager,
    new_history_kv_data: torch.Tensor,  # [num_layers, 2, num_tokens, num_heads, head_dim]
    new_history_kv_lengths: torch.Tensor,
    batch_size: int,
    num_layers: int):

    new_history_kv_length_offsets = get_offsets_from_lengths(new_history_kv_lengths)

    page_size = gpu_kvcache_manager.tokens_per_block
    
    kv_page_indices = list()
    kv_page_indptr = list([0])
    kv_last_page_len = list()

    for seq_idx in range(batch_size):
        user_id = seq_idx
        new_history_length = new_history_kv_lengths[seq_idx].item()

        user_kv_data = new_history_kv_data[:, :, new_history_kv_length_offsets[seq_idx]:new_history_kv_length_offsets[seq_idx+1], ...]

        # Allocation
        gpu_kvcache_manager.impl.add_sequence_with_eviction(user_id, new_history_length, 1, None)
        # Copy data
        page_ids = gpu_kvcache_manager.impl.get_cache_block_ids(user_id)[0]
        print(user_id, page_ids)
        for layer_idx in range(num_layers):
            
            last_page_size = new_history_length % page_size
            for page_idx in range(0, (new_history_length-last_page_size) // page_size):
                page_id = page_ids[page_idx]
                token_begin = page_idx * page_size
                token_end = (page_idx + 1) * page_size
                gpu_kvcache_manager.get_buffers(layer_idx)[page_id, ... ].copy_(
                    user_kv_data[layer_idx, :, token_begin:token_end, ...], non_blocking=True)
            if last_page_size > 0:
                page_idx = (new_history_length-last_page_size) // page_size
                page_id = page_ids[page_idx]
                token_begin = page_idx * page_size
                token_end = token_begin + last_page_size
                gpu_kvcache_manager.get_buffers(layer_idx)[page_id, ... ] *= 0.0
                gpu_kvcache_manager.get_buffers(layer_idx)[page_id, :, :last_page_size, ... ].copy_(
                    user_kv_data[layer_idx, :, token_begin:token_end, ...], non_blocking=True)
        
        kv_page_indices.extend(page_ids)
        kv_page_indptr.append(kv_page_indptr[-1] + len(page_ids))
        kv_last_page_len.append( last_page_size if last_page_size > 0 else page_size )
    
    return (kv_page_indices, kv_page_indptr, kv_last_page_len)

def gather_kvdata_from_cache(kv_cache, page_ids, page_offsets, last_page_lens):
    
    num_heads = kv_cache.shape[-2]
    attention_dim = kv_cache.shape[-1]
    k_con = torch.empty((0, num_heads, attention_dim), device=kv_cache.device, dtype=kv_cache.dtype)
    v_con = torch.empty((0, num_heads, attention_dim), device=kv_cache.device, dtype=kv_cache.dtype)

    for i in range(len(last_page_lens)):
        page_num = page_offsets[i + 1] - page_offsets[i]
        # new_history_len = q_offsets[i + 1] - q_offsets[i] - num_targets[i]
        for j in range(page_num - 1):
            k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i] + j], 0, :, :, :]), dim=0)
            v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i] + j], 1, :, :, :]), dim=0)
        k_con = torch.cat((k_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 0, :last_page_lens[i], :, :]), dim=0)
        v_con = torch.cat((v_con, kv_cache[page_ids[page_offsets[i + 1] - 1], 1, :last_page_lens[i], :, :]), dim=0)
    
    return k_con, v_con

def get_reference_output(hstu_layers, batch_size, hidden_states, jd, kvcache_metadata):
    input_tensor = hidden_states
    for hstu_layer in hstu_layers:
        normed_input = F.layer_norm(
            input_tensor,
            normalized_shape=[hstu_layer._embedding_dim],
            weight=hstu_layer._input_layernorm_weight,
            bias=hstu_layer._input_layernorm_bias,
            eps=hstu_layer._eps,
        )
        
        mixed_uvqk = F.silu(hstu_layer._linear_uvqk(normed_input))
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            hstu_layer._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, hstu_layer._num_heads, hstu_layer._linear_dim_per_head)
        query = query.view(-1, hstu_layer._num_heads, hstu_layer._attention_dim_per_head)
        key = key.view(-1, hstu_layer._num_heads, hstu_layer._attention_dim_per_head)

        kv_cache_table = kvcache_metadata.kv_cache_table[hstu_layer.layer_idx]
        (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
        # print("nnz_cuda", kvcache_metadata.new_history_nnz_cuda)
        append_kvcache.forward(
            key, value,
            kvcache_metadata.batch_indices,
            kvcache_metadata.position,
            jd.num_candidates_offsets,
            kvcache_metadata.new_history_nnz_cuda,
            4096,
            paged_k_cache, paged_v_cache,
            kvcache_metadata.kv_indices,
            kvcache_metadata.kv_indptr,
            kvcache_metadata.kv_last_page_len,
            0
        )

        total_k_seqlen = kvcache_metadata.total_history_offsets[1:batch_size+1] - kvcache_metadata.total_history_offsets[:batch_size]
        mask = torch.zeros((batch_size, hstu_layer._num_heads, jd.seqlen[:batch_size].max().item(), total_k_seqlen.max().item()),
                           dtype=torch.bfloat16, device=hidden_states.device)
        for seq_idx in range(batch_size):
            qlen = jd.seqlen[seq_idx].item()
            num_cand = jd.num_candidates[seq_idx].item()
            cachelen = total_k_seqlen[seq_idx].item() - num_cand
            seq_mask = torch.cat([
                    torch.tril(torch.ones((qlen, cachelen), dtype=torch.int32), diagonal=cachelen+num_cand-qlen),
                    torch.cat([torch.zeros((qlen-num_cand, num_cand), dtype=torch.int32), torch.eye(num_cand, dtype=torch.int32)], dim=0)
                ], dim=1)
            mask[seq_idx, :, :qlen, :cachelen+num_cand].copy_(seq_mask.type(torch.bfloat16))

        ref_attn_output = _hstu_paged_kv_attention(
            num_heads=hstu_layer._num_heads,
            attention_dim=hstu_layer._attention_dim_per_head,
            linear_dim=hstu_layer._linear_dim_per_head,
            seqlen_q=4096,
            seqlen_k=4096,
            q=query,
            k=key,
            v=value,
            q_offsets=jd.seqlen_offsets[:batch_size+1],
            k_offsets=kvcache_metadata.total_history_offsets[:batch_size+1],
            num_targets=jd.num_candidates[:batch_size],
            invalid_attn_mask=mask,
            alpha=hstu_layer._alpha,
            upcast=False,
            kv_cache=kvcache_metadata.kv_cache_table[hstu_layer.layer_idx],
            page_offsets=kvcache_metadata.kv_indptr[:batch_size+1],
            page_ids=kvcache_metadata.kv_indices,
            last_page_lens=kvcache_metadata.kv_last_page_len[:batch_size]
        )

        ref_attn_output = ref_attn_output.view(-1, hstu_layer._num_heads*hstu_layer._linear_dim_per_head)
        parallel_input = user * F.layer_norm(
            ref_attn_output,
            normalized_shape=[hstu_layer._num_heads*hstu_layer._linear_dim_per_head],
            weight=hstu_layer._output_layernorm_weight,
            bias=hstu_layer._output_layernorm_bias,
            eps=hstu_layer._eps,
        )

        layer_output = hstu_layer._linear_proj(parallel_input)
        if hstu_layer._residual:
            layer_output = layer_output + input_tensor

        input_tensor = layer_output

    return layer_output


@pytest.mark.parametrize("model_type", ["ranking"])
@pytest.mark.parametrize("batchsize", [16])
@pytest.mark.parametrize("max_contextual_seqlen", [0])
@pytest.mark.parametrize(
    "item_max_seqlen,max_num_candidates",
    [
        # (2, 10),
        # (10, 6)

        (128, 128),         # 512
        # (128*3, 128),     # 1024
        # (512, 128),       # 1024 + 256
        # (512+128*3, 128), # 2048
        # (1024, 128),      # 2048 + 256
    ],
)
@pytest.mark.parametrize("dim_size", [128,])
def test_gr_infer(
    model_type,
    batchsize,
    item_max_seqlen,
    dim_size,
    max_num_candidates,
    max_contextual_seqlen,
):
    init.initialize_single_rank()
    init.initialize_model_parallel(1)
    init.set_random_seed(1234)
    device = torch.cuda.current_device()

    hstu_config = get_hstu_config(
        hidden_size=dim_size,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=8,
        dtype=torch.bfloat16,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=51200,
        tokens_per_block=32,
        max_batch_size=batchsize,
        max_seq_len=4096,
    )

    context_emb_size = 1000
    item_emb_size = 1000
    action_vocab_size = 10
    num_tasks = 1
    # embedding_optimizer_param = EmbeddingOptimizerParam(
    #     optimizer_str="adam", learning_rate=0.0001
    # )
    emb_configs = [
        ShardedEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=dim_size,
            sharding_type="data_parallel",
        ),
        ShardedEmbeddingConfig(
            feature_names=["context_feat", "item_feat"]
            if max_contextual_seqlen > 0
            else ["item_feat"],
            table_name="item",
            vocab_size=item_emb_size,
            dim=dim_size,
            sharding_type="model_parallel",
        ),
    ]
    feature_configs = [
        FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[item_emb_size, action_vocab_size],
            max_sequence_length=item_max_seqlen + max_num_candidates,
            is_jagged=False,
        ),
    ]
    if max_contextual_seqlen > 0:
        feature_configs.append(
            FeatureConfig(
                feature_names=["context_feat"],
                max_item_ids=[context_emb_size],
                max_sequence_length=10,
                is_jagged=True,
            ),
        )
    batch_kwargs = dict(
        batch_size=batchsize,
        feature_configs=feature_configs,
        item_feature_name="item_feat",
        contextual_feature_names=["context_feat"] if max_contextual_seqlen > 0 else [],
        action_feature_name="act_feat",
        max_num_candidates=max_num_candidates,
        device=device,
    )
    if model_type == "ranking":
        task_config = RankingConfig(
            embedding_configs=emb_configs,
            prediction_head_arch=[[128, 10, 1] for _ in range(num_tasks)],
        )
        model_predict = RankingGRInferenceModel(
            hstu_config=hstu_config, 
            kvcache_config=kv_cache_config, 
            task_config=task_config,
            use_cudagraph=True)
        model_predict.bfloat16()
        model_predict.eval()

    with torch.inference_mode():

        user_ids_list = [ torch.arange(i * batchsize, (i + 1) * batchsize) for  i in range(2) ]
        batch_list = [ RankingBatch.random(num_tasks=num_tasks, **batch_kwargs) for _ in range(2) ]
        num_tokens_list = [ batch.features.values().shape[0] for batch in batch_list ]

        jagged_data_list = [ 
            model_predict._hstu_block.hstu_preprocess(
                embeddings=model_predict._embedding_collection(batch.features),
                batch=batch)
            for batch in batch_list ]
        input_list = [ jd.values for jd in jagged_data_list ]
        
        kvcache_metadata_list = []
        for i in range(len(jagged_data_list)):
            jagged_data_list[i].values.copy_(torch.randn_like(jagged_data_list[i].values))
            kvcache_metadata_list.append(model_predict.prepare_kv_cache(batch_list[i], user_ids_list[i]))
        
        num_warmup = 10
        num_runs = 1000

        torch.cuda.synchronize()
        for run_iter in range(num_warmup):
            num_tokens = num_tokens_list[run_iter%2]
            input_data = input_list[run_iter%2]
            jagged_data = jagged_data_list[run_iter%2]
            kvcache_metadata = kvcache_metadata_list[run_iter%2]

            model_predict._hidden_states[:num_tokens, ...].copy_(input_data, non_blocking=True)
            copy_jagged_metadata(model_predict._jagged_metadata, jagged_data)
            copy_kvcache_metadata(model_predict._kvcache_metadata, kvcache_metadata)
            model_predict._kvcache_metadata.total_history_offsets += model_predict._jagged_metadata.num_candidates_offsets
            
            hstu_output = model_predict._hstu_block.predict(batchsize, num_tokens, model_predict._hidden_states, model_predict._jagged_metadata, model_predict._kvcache_metadata)
        
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for run_iter in range(num_runs):
            num_tokens = num_tokens_list[run_iter%2]
            input_data = input_list[run_iter%2]
            jagged_data = jagged_data_list[run_iter%2]
            kvcache_metadata = kvcache_metadata_list[run_iter%2]

            # model_predict._hidden_states[:num_tokens, ...].copy_(input_data, non_blocking=True)
            copy_jagged_metadata(model_predict._jagged_metadata, jagged_data)
            copy_kvcache_metadata(model_predict._kvcache_metadata,kvcache_metadata)
            model_predict._kvcache_metadata.total_history_offsets += model_predict._jagged_metadata.num_candidates_offsets
            
            hstu_output = model_predict._hstu_block.predict(batchsize, num_tokens, model_predict._hidden_states, model_predict._jagged_metadata, model_predict._kvcache_metadata)
        
        stop.record()
        torch.cuda.synchronize()
        print("[cudagraph] Time (ms):", start.elapsed_time(stop) / float(num_runs))

        for i in range(2):
            kvcache_metadata_list[i].total_history_offsets += jagged_data_list[i].num_candidates_offsets
        use_cudagraph = False
        model_predict.use_cudagraph = False

        torch.cuda.synchronize()
        for run_iter in range(num_warmup):
            num_tokens = num_tokens_list[run_iter%2]
            input_data = input_list[run_iter%2]
            jagged_data = jagged_data_list[run_iter%2]
            kvcache_metadata = kvcache_metadata_list[run_iter%2]

            # model_predict._hidden_states[:num_tokens, ...].copy_(input_data, non_blocking=True)
            hstu_output_2 = model_predict._hstu_block.predict(batchsize, num_tokens, input_data, jagged_data, kvcache_metadata, use_cudagraph)
            # model_predict._hidden_states[:num_tokens, ...].copy_(hstu_output_2, non_blocking=True)

        torch.cuda.synchronize()
        start.record()
        for run_iter in range(num_runs):
            num_tokens = num_tokens_list[run_iter%2]
            input_data = input_list[run_iter%2]
            jagged_data = jagged_data_list[run_iter%2]
            kvcache_metadata = kvcache_metadata_list[run_iter%2]

            # model_predict._hidden_states[:num_tokens, ...].copy_(input_data, non_blocking=True)
            hstu_output_2 = model_predict._hstu_block.predict(batchsize, num_tokens, input_data, jagged_data, kvcache_metadata, use_cudagraph)
            # model_predict._hidden_states[:num_tokens, ...].copy_(hstu_output_2, non_blocking=True)

        stop.record()
        torch.cuda.synchronize()
        print("[vallina] Time (ms):", start.elapsed_time(stop) / float(num_runs))

        print(hstu_output_2.abs().mean().item(), hstu_output_2.abs().max().item())
        print(hstu_output.abs().mean().item(), hstu_output.abs().max().item())

    init.destroy_global_state()

    

