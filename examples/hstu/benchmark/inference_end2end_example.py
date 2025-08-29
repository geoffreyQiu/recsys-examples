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
import sys
import time

from commons.utils.stringify import stringify_dict
from dataclasses import dataclass
import gin
import torch
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.random_inference_dataset import RandomInferenceDataGenerator
from dataset.utils import FeatureConfig
import math
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from typing import List, Tuple, cast

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR
sys.path.append("./training/")
from gin_config_args import TrainerArgs, EmbeddingArgs, DatasetArgs, NetworkArgs, OptimizerArgs

#duplicate
@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"

def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=233,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return dataset_args, embedding_configs if not disable_contextual_features else embedding_configs[-2:]
    
    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")


def run_ranking_gr_inference(
    checkpoint_dir: str,
    check_auc: bool = False,
    disable_contextual_features: bool = False
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(disable_contextual_features)
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    elif network_args.dtype_str == "float16":
        inference_dtype = torch.float16
    else:
        raise ValueError(f"Inference data type {network_args.dtype_str} is not supported")

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    dataproc._batching_file = dataproc._output_file.replace("processed_seqs", "processed_batches")
    num_contextual_features = len(dataproc._contextual_feature_names) if not disable_contextual_features else 0

    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 512,
        "page_size": 32,
        "offload_chunksize": 32,
        "max_batch_size": 1,
        "max_seq_len": math.ceil(total_max_seqlen / 32) * 32,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    # TODO: need to init & setup
    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    with torch.inference_mode():
        model_predict = InferenceRankingGR(
            hstu_config=hstu_config,
            kvcache_config=kv_cache_config,
            task_config=task_config,
            use_cudagraph=True,
            cudagraph_configs=hstu_cudagraph_configs,
        )
        if hstu_config.bf16:
            model_predict.bfloat16()
        elif hstu_config.fp16:
            model_predict.half()
        model_predict.load_checkpoint(checkpoint_dir)
        model_predict.eval()

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=task_config.prediction_head_arch[-1],
                num_tasks=task_config.num_tasks,
                metric_types=task_config.eval_metrics,
            )

        dataset = InferenceDataset(
            seq_logs_file=dataproc._output_file,
            batch_logs_file=dataproc._batching_file,
            batch_size=kvcache_args["max_batch_size"],
            max_seqlen=dataset_args.max_sequence_length,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names if not disable_contextual_features else [],
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,

            item_vocab_size=10_000_000,
            userid_name='user_id',
            date_name='date',
            sequence_endptr_name='interval_indptr',
            timestamp_names=['date', 'interval_end_ts'],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        seq_lengths = set()
        seqlen_histogram = dict()

        # ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start_time = time.time()
        cur_date = None
        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
                if dates[0] != cur_date:
                    if cur_date is not None:
                        eval_metric_dict = eval_module.compute()
                        print(
                            f"[eval]:\n    "
                            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
                        )
                    model_predict.clear_kv_cache()
                    cur_date = dates[0]
                cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids, dbg_print=True)
                new_cache_start_pos = cached_start_pos + cached_len
                non_contextual_mask = new_cache_start_pos >= num_contextual_features
                contextual_mask = torch.logical_not(non_contextual_mask)
                seq_startptrs = (torch.clip(new_cache_start_pos - num_contextual_features, 0) / 2).int()

                batch_0 = dataset.get_input_batch(
                    uids[non_contextual_mask],
                    dates[non_contextual_mask],
                    seq_endptrs[non_contextual_mask],
                    seq_startptrs[non_contextual_mask], 
                    with_contextual_features = False, 
                    with_ranking_labels = True)
                if batch_0 is not None:
                    logits = model_predict.forward(batch_0, uids[non_contextual_mask].int(), new_cache_start_pos[non_contextual_mask])
                    eval_module(logits, batch_0.labels)

                batch_1 = dataset.get_input_batch(
                    uids[contextual_mask],
                    dates[contextual_mask],
                    seq_endptrs[contextual_mask],
                    seq_startptrs[contextual_mask], 
                    with_contextual_features = True, 
                    with_ranking_labels = True)
                if batch_1 is not None:
                    logits = model_predict.forward(batch_1, uids[contextual_mask].int(), new_cache_start_pos[contextual_mask])
                    eval_module(logits, batch_1.labels)

                num_batches_ctr += 1
            except StopIteration:
                break
        end_time = time.time()
        print("Total #batch:", num_batches_ctr)
        print("Total time(s):", end_time-start_time)

if __name__ == "__main__":
    # TODO: change to args (argsparser)
    gin.parse_config_file("./kuairand_1k_inference_ranking.gin")
    checkpoint_dir = '/home/junyiq//newscratch/june/kuairand1k_checkpoints/iter1500/'
    check_auc = True
    run_ranking_gr_inference(
        checkpoint_dir = checkpoint_dir,
        check_auc = check_auc
    )
