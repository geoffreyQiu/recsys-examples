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
import copy

# pyre-strict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from configs.task_config import ShardedEmbeddingConfig as HstuEmbeddingConfig
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from torch import distributed as dist
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.distributed.utils import (
    add_params_from_parameter_sharding,
    convert_to_fbgemm_types,
    merge_fused_params,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    EmbeddingTableConfig,
    PoolingType,
    dtype_to_data_type,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from dynamicemb import (
    DynamicEmbTableOptions, 
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables

from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)


class ParameterServer(torch.nn.Module):
    pass

class DummyParameterServer(ParameterServer):
    def __init__(self, embedding_configs):
        super().__init__()
        self._embedding_collection = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=config.table_name,
                    embedding_dim=config.dim,
                    num_embeddings=config.vocab_size,
                    feature_names=config.feature_names,
                    data_type=dtype_to_data_type(torch.float32),
                )
                for config in embedding_configs
            ],
            device=torch.device("meta"),
        )
    
    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        return self._embedding_collection(features)
    

def create_dynamic_embedding_tables(
    embedding_configs: List[HstuEmbeddingConfig],
    output_dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    ps: ParameterServer = None,
):

    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=config.dim,
            max_capacity=config.vocab_size,

            local_hbm_for_values=0,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
            ),
        )
        for config in embedding_configs
    ]

    table_names = [ config.table_name for config in embedding_configs ]

    return BatchedDynamicEmbeddingTables(
        table_options=table_options,
        table_names=table_names,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=output_dtype,
    )

class InferenceDynamicEmbeddingCollection(nn.Module):
    def __init__(self, embedding_configs, ps: ParameterServer = None, enable_cache: bool = False):
        super().__init__()
        
        self._embedding_tables = create_dynamic_embedding_tables(
            embedding_configs,
            ps = ps)

        self._cache = create_dynamic_embedding_tables(
            embedding_configs,
            device = torch.cuda.current_device()
        ) if enable_cache else None

        self._feature_names = [ feature for config in embedding_configs for feature in config.feature_names ]

        self._has_uninitialized_input_dist = True
        self._features_order: List[int] = []
        self.register_buffer(
            "_features_order_tensor",
            torch.zeros((len(self._feature_names)), device=torch.cuda.current_device(), dtype=torch.int32),
            persistent=False,
        )
    
    def get_input_dist(
        self,
        input_feature_names: List[str],
    ) -> int:
        input_features_order = []
        for f in self._feature_names:
            if f in input_feature_names:
                input_features_order.append(input_feature_names.index(f))
        
        num_input_features = len(input_features_order)
            
        if self._has_uninitialized_input_dist or input_features_order != self._features_order:
            self._features_order = (
                []
                if input_features_order == list(range(num_input_features))
                else input_features_order
            )
            if len(self._features_order) > 0:
                self._features_order_tensor[:num_input_features]._copy(
                    torch.tensor(input_features_order, device=torch.cuda.current_device(), dtype=torch.int32))
        
        if self._has_uninitialized_input_dist:
            self._has_uninitialized_input_dist = False

        return num_input_features

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        num_input_features = self.get_input_dist(input_feature_names=features.keys())
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor[:len(self._features_order)],
                )
            features = features.split([num_input_features])[0]
            embeddings = self._embedding_tables(features.values(), features.offsets())
        embeddings_kjt = KeyedJaggedTensor(
            values=embeddings,
            keys=features.keys(),
            lengths=features.lengths(),
            offsets=features.offsets(),
        )
        return embeddings_kjt.to_dict()

def create_embedding_collection(configs):
    return EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=config.table_name,
                embedding_dim=config.dim,
                num_embeddings=config.vocab_size,
                feature_names=config.feature_names,
                data_type=dtype_to_data_type(torch.float32),
            )
            for config in configs
        ],
        device=torch.device("meta"),
    )

def create_dynamic_embedding_collection(configs, ps: ParameterServer = None, enable_cache: bool = False):
    return InferenceDynamicEmbeddingCollection(configs, ps, enable_cache)


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[HstuEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
    """

    def __init__(
        self,
        embedding_configs: List[HstuEmbeddingConfig],
    ):
        super(InferenceEmbedding, self).__init__()

        model_parallel_embedding_configs = []
        data_parallel_embedding_configs = []
        for config in embedding_configs:
            if config.sharding_type == "data_parallel":
                data_parallel_embedding_configs.append(config)
            else:
                model_parallel_embedding_configs.append(config)

        self._model_parallel_embedding_collection = create_dynamic_embedding_collection(
            configs=model_parallel_embedding_configs,
            ps=None, enable_cache=False)

        if len(data_parallel_embedding_configs) > 0:
            self._data_parallel_embedding_collection = create_embedding_collection(
                configs=data_parallel_embedding_configs
            )
            self._side_stream = torch.cuda.Stream()
        else:
            self._data_parallel_embedding_collection = None

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """
        mp_embeddings = self._model_parallel_embedding_collection(kjt)
        if self._data_parallel_embedding_collection is not None:
            with torch.cuda.stream(self._side_stream):
                dp_embeddings = self._data_parallel_embedding_collection(kjt)
            for k in mp_embeddings:
                print("key", k, mp_embeddings[k].values().shape, mp_embeddings[k].values().dtype)
            for k in dp_embeddings:
                print("key", k, dp_embeddings[k].values().shape, dp_embeddings[k].values().dtype)
            torch.cuda.current_stream().wait_stream(self._side_stream)
            embeddings = {**mp_embeddings, **dp_embeddings}
        else:
            embeddings = mp_embeddings
        return embeddings


def get_nonfused_embedding_optimizer(
    module: torch.nn.Module,
) -> Iterator[torch.optim.Optimizer]:
    """
    Retrieves non-fused embedding optimizers from a PyTorch module. Non-fused embedding optimizers are used by torchrec data-parallel sharded embedding collection.

    Args:
        module (torch.nn.Module): The PyTorch module to search for non-fused embedding optimizers.

    Yields:
        torch.optim.Optimizer: An iterator over the non-fused embedding optimizers found in the module.
    """
    for module in module.modules():
        if hasattr(module, "_nonfused_embedding_optimizer"):
            yield module._nonfused_embedding_optimizer
