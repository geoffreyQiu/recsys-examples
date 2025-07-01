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
)
from dataset.utils import FeatureConfig, RankingBatch, RetrievalBatch
from megatron.core import tensor_parallel
from model.ranking_gr_infer import RankingGRInferenceModel, copy_jagged_metadata
from modules.jagged_data import JaggedData
import torch.nn.functional as F


def get_offsets_from_lengths(lengths):
    offsets = torch.zeros((lengths.shape[0]+1,), dtype=lengths.dtype, device=lengths.device)
    torch.cumsum(lengths, 0, out=offsets[1:])
    return offsets

def get_reference_output(
    hstu_block: torch.nn.Module,
    batch_size: int,
    hidden_states: torch.Tensor, 
    kvcache_data: torch.Tensor,  # [num_layers, 2, num_tokens, num_heads, head_dim]
    kvcache_data_lengths: torch.Tensor,
    jagged_metadata: JaggedData,
    kvcache_metadata: KVCacheMetadata):

    kvcache_data_offsets = torch.cat([ torch.zeros((1,), dtype=torch.int32), torch.cumsum(kvcache_data_lengths, dim = 0) ], dim = 0)

    input_tensor = hidden_states
    for hstu_layer in hstu_block._attention_layers:
        # print(hstu_layer._linear_uvqk.weight.shape)
        # print(hstu_layer._linear_uvqk.bias.shape)
        # print(hstu_layer._linear_proj.weight.shape)
        # print("[DEBUG]", jagged_metadata.seqlen_offsets.cpu())

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

        seq_output_list = list()
        for seq_idx in range(batch_size):
            seq_token_start = jagged_metadata.seqlen_offsets[seq_idx].item()
            seq_token_end = jagged_metadata.seqlen_offsets[seq_idx + 1].item()

            cache_token_start = kvcache_data_offsets[seq_idx].item()
            cache_token_end = kvcache_data_offsets[seq_idx + 1].item()

            seq_token_cand = seq_token_end - jagged_metadata.num_candidates[seq_idx].item()

            head_output_list = list()
            for head_idx in range(hstu_layer._num_heads):
                if seq_token_cand > seq_token_start:
                    if cache_token_end > cache_token_start:
                        o_hist = torch.matmul(
                            torch.matmul(query[seq_token_start:seq_token_cand, head_idx, ...],
                                            kvcache_data[hstu_layer.layer_idx, 0, cache_token_start:cache_token_end, head_idx, ...].T),
                            kvcache_data[hstu_layer.layer_idx, 1, cache_token_start:cache_token_end, head_idx, ...])
                    else:
                        o_hist = 0
            
                    o_hist = o_hist + torch.matmul(
                        torch.tril(torch.matmul(query[seq_token_start:seq_token_cand, head_idx, ...], 
                                                key[seq_token_start:seq_token_cand, head_idx, ...].T)),
                        value[seq_token_start:seq_token_cand, head_idx, ...])
                else:
                    o_hist = None
                
                if seq_token_end > seq_token_cand:
                    if cache_token_end > cache_token_start:
                        o_cand = torch.matmul(
                            torch.matmul(query[seq_token_cand:seq_token_end, head_idx, ...],
                                         kvcache_data[hstu_layer.layer_idx, 0, cache_token_start:cache_token_end, head_idx, ...].T),
                            kvcache_data[hstu_layer.layer_idx, 1, cache_token_start:cache_token_end, head_idx, ...])
                    else:
                        o_cand = 0
                    
                    if seq_token_cand > seq_token_start:
                        o_cand = o_cand + torch.matmul(
                            torch.matmul(query[seq_token_cand:seq_token_end, head_idx, ...], 
                                        key[seq_token_cand:seq_token_end, head_idx, ...].T),
                            value[seq_token_cand:seq_token_end, head_idx, ...])
                    
                    o_cand = o_cand + torch.matmul(
                        torch.matmul(query[seq_token_cand:seq_token_end, head_idx, ...], 
                                     key[seq_token_cand:seq_token_end, head_idx, ...].T)
                        * torch.eye(seq_token_end-seq_token_cand, dtype=query.dtype, device=query.device),
                        value[seq_token_cand:seq_token_end, head_idx, ...])
                else:
                    o_cand = None
                
                if o_hist is not None and o_cand is not None:
                    head_output = torch.cat([o_hist, o_cand], dim = 0)
                elif o_hist is None:
                    head_output = o_cand
                elif o_cand is None:
                    head_output = o_hist

                head_output_list.append(head_output)
            seq_output = torch.cat(head_output_list, dim=1)
            seq_output_list.append(seq_output)
        attn_output = torch.cat(seq_output_list, dim=0)
        print("attn_output.shape", attn_output.shape)

        parallel_input = user * F.layer_norm(
            attn_output,
            normalized_shape=[hstu_layer._num_heads*hstu_layer._linear_dim_per_head],
            weight=hstu_layer._output_layernorm_weight,
            bias=hstu_layer._output_layernorm_bias,
            eps=hstu_layer._eps,
        )

        output_tensor = hstu_layer._linear_proj(parallel_input)
        if hstu_layer._residual:
            output_tensor = output_tensor + input_tensor
        
        input_tensor = output_tensor
    
    return input_tensor

    ''' '''
    ### for seq_idx in range(batch_size):
    ###     # print("---")
    ###     # print(kv_last_page_len, new_history_kv_lengths)
    ###     if new_history_kv_lengths[seq_idx] == 0:
    ###         # print("---")
    ###         continue
    ###   
    ###     last_page_len = kv_last_page_len[seq_idx].item()
    ###     # previous page:
    ###     token_offset = jagged_metadata.seqlen_offsets[seq_idx].item()
    ###     if new_history_kv_lengths[seq_idx] > last_page_len:
    ###         length = new_history_kv_lengths[seq_idx] - last_page_len
    ###         start_page_idx = -math.ceil(length / 32) - 1
    ###         if length % 32 > 0:
    ###             length = length % 32
    ###             page_id = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[seq_idx+1]+start_page_idx]

    ###             kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 0, 32-length:32, ...].copy_(
    ###                 key[token_offset:token_offset+length, ...])
    ###             kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 1, 32-length:32, ...].copy_(
    ###                 value[token_offset:token_offset+length, ...])
    ###             print(length, ":", token_offset, token_offset+length, ":", page_id.item(), ",", 32-length, 32 )
    ###           
    ###             token_offset += length
    ###             start_page_idx += 1
    ###       
    ###         for page_idx in range(start_page_idx, -1):
    ###             page_id = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[seq_idx+1]+page_idx]
    ###             kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 0, ...].copy_(
    ###                 key[token_offset:token_offset+32, ...])
    ###             kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 1, ...].copy_(
    ###                 value[token_offset:token_offset+32, ...])
    ###             print(32, ":", token_offset, token_offset+32, ":", page_id.item(), ",", 0, 32 )
    ###           
    ###             token_offset += 32

    ###     # last page
    ###     length = min(new_history_kv_lengths[seq_idx], last_page_len)
    ###     page_id = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[seq_idx+1]-1]
    ###     kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 0, last_page_len-length:last_page_len, ...].copy_(
    ###         key[token_offset:token_offset+length, ...])
    ###     kvcache_metadata.kv_cache_table[hstu_layer.layer_idx][page_id, 1, last_page_len-length:last_page_len, ...].copy_(
    ###         value[token_offset:token_offset+length, ...])
    ###     # print(length, ":", token_offset, token_offset+length, ":", page_id.item(), ",", last_page_len-length, last_page_len )
    ###     # print("-+-")

    ''' '''


def generate_kvdata_testcase(
    max_seq_len: int,
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int):

    lengths = torch.randint(max_seq_len, (batch_size,), dtype=torch.int32)
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
        
        kv_page_indices.extend(page_ids)
        kv_page_indptr.append(kv_page_indptr[-1] + len(page_ids))
        kv_last_page_len.append( last_page_size if last_page_size > 0 else page_size )
    
    return (kv_page_indices, kv_page_indptr, kv_last_page_len)

@pytest.mark.parametrize("model_type", ["ranking"])
@pytest.mark.parametrize("batchsize_per_rank", [4])
@pytest.mark.parametrize("max_contextual_seqlen", [0])
@pytest.mark.parametrize(
    "item_max_seqlen,max_num_candidates",
    [
        # (2, 10),
        (20, 10),
        # (200, 10),
    ],
)
@pytest.mark.parametrize("dim_size", [128,])
def test_gr_infer(
    model_type,
    batchsize_per_rank,
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
        num_layers=3,
        dtype=torch.bfloat16,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=51200,
        tokens_per_block=32,
        max_batch_size=batchsize_per_rank,
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
            is_jagged=True,
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
        batch_size=batchsize_per_rank,
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
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = RankingBatch.random(num_tasks=num_tasks, **batch_kwargs)
        user_ids = torch.arange(batchsize_per_rank)
    
    kv_data, kv_data_lengths = generate_kvdata_testcase(4096, batchsize_per_rank, hstu_config.num_layers, hstu_config.num_attention_heads, hstu_config.kv_channels)
    kv_raw_metadata = setup_kvcache_testcase(model_predict._gpu_kv_cache_manager, kv_data, kv_data_lengths, batchsize_per_rank, hstu_config.num_layers)
    print("kvcache", kv_raw_metadata[1], kv_raw_metadata[2])

    # output_logit = model_predict(batch, user_ids)
    with torch.inference_mode():
        kvcache_metadata = model_predict.prepare_kv_cache(batch, user_ids)
        print("kvcache", model_predict._kvcache_metadata.kv_indptr, model_predict._kvcache_metadata.kv_last_page_len)

        jagged_data = model_predict._hstu_block.hstu_preprocess(
            embeddings=model_predict._embedding_collection(batch.features),
            batch=batch,
        )

        print(">>>")
        print(jagged_data.values.shape)
        print(batch.features.values().shape)
        print("max_seqlen", jagged_data.max_seqlen)
        print("seq_lengths", jagged_data.seqlen)
        print("seq_lengths_offsets", jagged_data.seqlen_offsets)
        print("num_candidates", jagged_data.num_candidates)
        print("num_candidates_offsets", jagged_data.num_candidates_offsets)
        print("<<<")

        print("==========")
        print(model_predict._hidden_states[:batch.features.values().shape[0], ...].shape, jagged_data.values.shape)
        print("==========")
        model_predict._hidden_states[:batch.features.values().shape[0], ...].copy_(jagged_data.values, non_blocking=True)

        copy_jagged_metadata(model_predict._jagged_metadata, jagged_data)
        hstu_output = model_predict._hstu_block.predict(batch.batch_size, jagged_data.seqlen_offsets[-1].item(), jagged_data.values, model_predict._jagged_metadata, model_predict._kvcache_metadata)

        hstu_output = hstu_output.values
        print("canary", hstu_output.shape)
        kvcache_metadata.kv_cache_table = model_predict._kvcache_metadata.kv_cache_table
        ref_hstu_output = get_reference_output(model_predict._hstu_block, batchsize_per_rank, jagged_data.values, 
                                               kv_data, kv_data_lengths,
                                               jagged_data, kvcache_metadata)

        model_predict.finalize_kv_cache(user_ids)

    init.destroy_global_state()
