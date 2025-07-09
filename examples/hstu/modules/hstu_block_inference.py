# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import itertools
import math
from typing import Dict, Optional

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from configs.hstu_config import HSTUConfig
from configs.kv_cache_config import KVCacheConfig
from dataset.utils import Batch
from megatron.core.transformer.module import MegatronModule
from modules.jagged_data import JaggedData
from modules.paged_hstu_infer_layer import PagedHSTUInferLayer
from modules.position_encoder import HSTUPositionalEncoder
from ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from ops.length_to_offsets import length_to_complete_offsets
from ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from torchrec.sparse.jagged_tensor import JaggedTensor


class HSTUBlockInference(MegatronModule):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: HSTUConfig,
        kvcache_config: KVCacheConfig = None,
    ):
        super().__init__(config=config)
        self._training_dtype = torch.float32
        if self.config.bf16:
            self._training_dtype = torch.bfloat16
        if self.config.fp16:
            self._training_dtype = torch.float16

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=True,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
            )
        self._attention_layers = torch.nn.ModuleList(
            [
                PagedHSTUInferLayer(config, kvcache_config, layer_idx)
                for layer_idx in range(self.config.num_layers)
            ]
        )
        self._hstu_graph = None

    @output_nvtx_hook(nvtx_tag="HSTUBlock preprocess", hook_key_or_attr_name="values")
    def hstu_preprocess(
        self, embeddings: Dict[str, JaggedTensor], batch: Batch
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings (candidates excluded).
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (Batch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        item_jt = embeddings[batch.item_feature_name]  # history + candidate
        sequence_embeddings = item_jt.values()
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_lengths_offsets = item_jt.offsets()
        sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

        if batch.action_feature_name is not None:
            action_jt = embeddings[batch.action_feature_name]
            sequence_embeddings.size(0)
            embedding_dim = sequence_embeddings.size(1)

            action_offsets = action_jt.offsets()
            item_offsets = item_jt.offsets()
            candidates_indptr = item_offsets[: batch.batch_size] + action_jt.lengths()

            item_embs, action_embs = item_jt.values(), action_jt.values()

            # pyre-ignore
            interleaved_embeddings = [(
                torch.cat([
                    item_embs[item_offsets[idx] : candidates_indptr[idx]],
                    action_embs[action_offsets[idx] : action_offsets[idx + 1]],
                ], dim=1).view(-1, embedding_dim),
                item_embs[candidates_indptr[idx] : item_offsets[idx + 1]],
            ) for idx in range(batch.batch_size) ]
            interleaved_embeddings = list(itertools.chain(*interleaved_embeddings))
            sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                -1, embedding_dim
            )
            sequence_embeddings_lengths = item_jt.lengths() + action_jt.lengths()
            sequence_embeddings_lengths_offsets = (
                item_jt.offsets() + action_jt.offsets()
            )
            sequence_max_seqlen += batch.feature_to_max_seqlen[
                batch.action_feature_name
            ]

        num_candidates = batch.num_candidates.cuda()
        max_num_candidates = batch.max_num_candidates

        contextual_max_seqlen = 0
        contextual_seqlen = None
        contextual_seqlen_offsets = None
        if len(batch.contextual_feature_names) > 0:
            contextual_max_seqlens = [
                batch.feature_to_max_seqlen[name]
                for name in batch.contextual_feature_names
            ]
            contextual_jts = [
                embeddings[name] for name in batch.contextual_feature_names
            ]
            all_values = [jt.values() for jt in contextual_jts] + [sequence_embeddings]
            all_offsets = [jt.offsets() for jt in contextual_jts] + [
                sequence_embeddings_lengths_offsets
            ]
            all_max_seqlens = contextual_max_seqlens + [sequence_max_seqlen]
            (
                sequence_embeddings,
                sequence_embeddings_lengths_after_concat,
            ) = jagged_2D_tensor_concat(
                all_values,
                all_offsets,
                all_max_seqlens,
            )
            contextual_max_seqlen = max(
                len(batch.contextual_feature_names), sum(contextual_max_seqlens)
            )

            contextual_seqlen = (
                sequence_embeddings_lengths_after_concat - sequence_embeddings_lengths
            )
            sequence_embeddings_lengths = sequence_embeddings_lengths_after_concat

            sequence_embeddings_lengths_offsets = (
                torch.ops.fbgemm.asynchronous_complete_cumsum(
                    sequence_embeddings_lengths
                )
            )

            contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                contextual_seqlen
            )

            sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

        if self._positional_encoder is not None:
            sequence_embeddings = self._positional_encoder(
                max_seq_len=sequence_max_seqlen,
                seq_lengths=sequence_embeddings_lengths,
                seq_offsets=sequence_embeddings_lengths_offsets,
                seq_timestamps=None,
                seq_embeddings=sequence_embeddings,
                num_targets=num_candidates,
            )

        return JaggedData(
            values=sequence_embeddings.to(self._training_dtype),
            seqlen=sequence_embeddings_lengths.to(
                torch.int32
            ),  # contextual + history + candidate
            seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
            max_seqlen=sequence_max_seqlen,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates.to(torch.int32)
            if num_candidates is not None
            else None,
            num_candidates_offsets=length_to_complete_offsets(num_candidates).to(
                torch.int32
            )
            if num_candidates is not None
            else None,
            contextual_max_seqlen=contextual_max_seqlen,
            contextual_seqlen=contextual_seqlen.to(torch.int32)
            if contextual_seqlen is not None
            else None,
            contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
            if contextual_seqlen_offsets is not None
            else None,
            has_interleaved_action=batch.action_feature_name is not None,
        )

    @output_nvtx_hook(nvtx_tag="HSTUBlock postprocess", hook_key_or_attr_name="values")
    def hstu_postprocess(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as candidates embedding for further processing.
        2. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """

        sequence_embeddings: torch.Tensor
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
            )
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
            )
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        sequence_embeddings = sequence_embeddings / torch.linalg.norm(
            sequence_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        return JaggedData(
            values=sequence_embeddings,
            seqlen=(seqlen_offsets[1:] - seqlen_offsets[:-1]).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
        )

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_key_or_attr_name="values")
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
        batch: Batch,
    ) -> JaggedData:
        """
        Forward pass of the HSTUBlock.

        Args:
            embeddings (Dict[str, JaggedTensor]): The input embeddings.
            batch (Batch): The input batch.

        Returns:
            JaggedData: The output jagged data.
        """
        jd = self.hstu_preprocess(embeddings, batch)
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        return self.hstu_postprocess(jd)

    @output_nvtx_hook(nvtx_tag="hstu_predict")
    def predict(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
        use_cudagraph: bool = True,
    ) -> torch.Tensor:
        if self._hstu_graph is None or not use_cudagraph:
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hidden_data = hstu_layer.forward_naive(
                    batch_size, num_tokens, hidden_data, jd, kv_cache_metadata
                )
            return hidden_data
        else:
            return self.predict_cudagraph(
                batch_size, num_tokens, hidden_states, kv_cache_metadata
            )

    def predict_naive(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> torch.Tensor:
        with torch.inference_mode():
            jagged_metadata = JaggedData(
                values=None,
                max_seqlen=jd.max_seqlen,
                seqlen=jd.seqlen[:batch_size],
                seqlen_offsets=jd.seqlen_offsets[: batch_size + 1],
                max_num_candidates=jd.max_num_candidates,
                num_candidates=jd.num_candidates[:batch_size],
                num_candidates_offsets=jd.num_candidates_offsets[: batch_size + 1],
                contextual_max_seqlen=jd.contextual_max_seqlen,
                contextual_seqlen=jd.contextual_seqlen,
                contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
                has_interleaved_action=jd.has_interleaved_action,
            )
            kv_cache_metadata.new_history_nnz = num_tokens
            hidden_data = hidden_states
            for hstu_layer in self._attention_layers:
                hstu_layer.forward_input(
                    batch_size,
                    num_tokens,
                    hidden_data,
                    jagged_metadata,
                    kv_cache_metadata,
                )
                hidden_data = hstu_layer.forward_output(
                    batch_size,
                    num_tokens,
                    hidden_data,
                    jagged_metadata,
                    kv_cache_metadata,
                )
            return hidden_data

    def predict_cudagraph(
        self,
        batch_size: int,
        num_tokens: int,
        hidden_states: torch.Tensor,
        kv_cache_metadata,
    ) -> torch.Tensor:
        with torch.inference_mode():
            batch_size = 2 ** math.ceil(math.log2(batch_size))
            if num_tokens not in self._hstu_graph[batch_size]:
                num_tokens_pow2 = max(32, 2 ** math.ceil(math.log2(num_tokens)))
            else:
                num_tokens_pow2 = num_tokens

            self._hstu_graph[batch_size][num_tokens_pow2][0].replay()
            for idx in range(1, self.config.num_layers + 1):
                kv_cache_metadata.onload_history_kv_events[idx - 1].wait(
                    torch.cuda.current_stream()
                )
                self._hstu_graph[batch_size][num_tokens_pow2][idx].replay()

            hstu_output = torch.zeros_like(hidden_states[:num_tokens, ...])
            hstu_output.copy_(
                self._attention_layers[-1].output_buffer_[:num_tokens, ...],
                non_blocking=True,
            )
            return hstu_output

    def set_cudagraph(
        self,
        max_batch_size,
        max_seq_len,
        static_hidden_states,
        static_jagged_metadata,
        static_kvcache_metadata,
    ):
        print("Setting up cuda graphs ...")
        if self._hstu_graph is None:
            self._hstu_graph = dict()
            self._hstu_graph[max_batch_size] = dict()

            torch.cuda.mem_get_info()[0]

            max_num_tokens = max_batch_size * max_seq_len
            graph_max = self.capture_graph(
                max_batch_size,
                max_num_tokens,
                static_hidden_states,
                static_jagged_metadata,
                static_kvcache_metadata,
                None,
            )
            self._hstu_graph[max_batch_size][max_num_tokens] = graph_max

            bs_list = [2**i for i in range(math.ceil(math.log2(max_batch_size)) + 1)]
            num_tokens_list = [
                2**i for i in range(5, math.ceil(math.log2(max_num_tokens)) + 1)
            ]

            for batch_size in bs_list:
                if batch_size not in self._hstu_graph:
                    self._hstu_graph[batch_size] = dict()
                for num_tokens in num_tokens_list:
                    if num_tokens // batch_size > max_seq_len:
                        break
                    if num_tokens in self._hstu_graph[batch_size]:
                        continue
                    self._hstu_graph[batch_size][num_tokens] = self.capture_graph(
                        batch_size,
                        num_tokens,
                        static_hidden_states,
                        static_jagged_metadata,
                        static_kvcache_metadata,
                        graph_max[0].pool(),
                    )

            torch.cuda.mem_get_info()[0]

    def capture_graph(
        self,
        batch_size,
        num_tokens,
        static_hidden_states,
        static_jagged_metadata,
        static_kvcache_metadata,
        memory_pool=None,
    ):
        torch.cuda.mem_get_info()[0]

        # Create CUDA stream
        graph_capture_warmup_stream = torch.cuda.Stream()
        graph_capture_warmup_stream.wait_stream(torch.cuda.current_stream())

        seqlen = num_tokens // batch_size
        static_jagged_metadata.seqlen_offsets[: batch_size + 1].copy_(
            torch.arange(
                end=batch_size + 1,
                dtype=static_jagged_metadata.num_candidates.dtype,
                device=static_jagged_metadata.num_candidates.device,
            )
            * seqlen
        )

        default_num_candidates = seqlen // 2
        torch.full(
            (batch_size,),
            default_num_candidates,
            out=static_jagged_metadata.num_candidates[:batch_size],
        )
        static_jagged_metadata.num_candidates_offsets[: batch_size + 1].copy_(
            torch.arange(
                end=batch_size + 1,
                dtype=static_jagged_metadata.num_candidates.dtype,
                device=static_jagged_metadata.num_candidates.device,
            )
            * default_num_candidates
        )

        static_kvcache_metadata.total_history_offsets += (
            static_jagged_metadata.num_candidates_offsets
        )
        static_kvcache_metadata.new_history_nnz = num_tokens

        # Warmup
        with torch.cuda.stream(graph_capture_warmup_stream):
            for _ in range(3):
                static_output = self.predict_naive(
                    batch_size,
                    num_tokens,
                    static_hidden_states,
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )
                torch.cuda.synchronize()

        # Create and capture the graph
        num_layers = self.config.num_layers
        graph = [torch.cuda.CUDAGraph() for _ in range(num_layers + 1)]
        input_buffer = [static_hidden_states] + [
            self._attention_layers[layer_idx].output_buffer_
            for layer_idx in range(num_layers)
        ]

        with torch.cuda.graph(graph[0], pool=memory_pool):
            static_uvqk = self._attention_layers[0].forward_input(
                batch_size,
                num_tokens,
                static_hidden_states,
                static_jagged_metadata,
                static_kvcache_metadata,
            )

        if memory_pool is None:
            memory_pool = graph[0].pool()

        for layer_idx in range(0, num_layers - 1):
            with torch.cuda.graph(graph[layer_idx + 1], pool=memory_pool):
                static_output = self._attention_layers[layer_idx].forward_output(
                    batch_size,
                    num_tokens,
                    input_buffer[layer_idx],
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )
                static_uvqk = self._attention_layers[layer_idx + 1].forward_input(
                    batch_size,
                    num_tokens,
                    static_output,
                    static_jagged_metadata,
                    static_kvcache_metadata,
                )

        with torch.cuda.graph(graph[num_layers], pool=memory_pool):
            static_output = self._attention_layers[-1].forward_output(
                batch_size,
                num_tokens,
                input_buffer[num_layers - 1],
                static_jagged_metadata,
                static_kvcache_metadata,
            )

        torch.cuda.mem_get_info()[0]

        static_kvcache_metadata.total_history_offsets -= (
            static_jagged_metadata.num_candidates_offsets
        )
        print(
            "Capture cuda graphs for batch_size = {0} and num_tokens = {1}".format(
                batch_size, num_tokens
            )
        )
        return graph
