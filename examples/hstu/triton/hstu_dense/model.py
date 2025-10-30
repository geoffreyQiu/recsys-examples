# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
from torch import nn

import numpy as np

TRITON_STRING_TO_NUMPY = {
    "TYPE_BOOL": torch.bool,
    "TYPE_UINT8": torch.uint8,
    "TYPE_UINT16": torch.uint16,
    "TYPE_UINT32": torch.uint32,
    "TYPE_UINT64": torch.uint64,
    "TYPE_INT8": torch.int8,
    "TYPE_INT16": torch.int16,
    "TYPE_INT32": torch.int32,
    "TYPE_INT64": torch.int64,
    "TYPE_BF16": torch.bfloat16,
    "TYPE_FP16": torch.float16,
    "TYPE_FP32": torch.float32,
    "TYPE_FP64": torch.float64,
}

def triton_string_to_torch_dtype(triton_type_string):
    return TRITON_STRING_TO_NUMPY[triton_type_string]

class TestModel(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, user_ids):
        torch.ones_like(user_ids)
        return hidden_state, num_cached_length, seqlen_offsets, num_candidates_offsets

class HSTUDenseNet(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(HSTUDenseNet, self).__init__()

    def forward(self, hidden_state, user_ids, seq_len, num_candidates):
        num_cached_length = user_ids
        seqlen_offsets = seq_len
        num_candidates_offsets = num_candidates
        return hidden_state, num_cached_length, seqlen_offsets, num_candidates_offsets


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        print("before:", torch.cuda.device_count())
        os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
        # print(args["model_instance_device_id"])
        print("after:", torch.cuda.device_count())

        # Get OUTPUT configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        # Get SEQ_OFFSETS configuration
        seq_offsets_config = pb_utils.get_output_config_by_name(model_config, "SEQ_OFFSETS")

        # Convert Triton types to numpy types
        self.output_dtype = triton_string_to_torch_dtype(
            output_config["data_type"]
        )
        self.seq_offsets_dtype = triton_string_to_torch_dtype(
            seq_offsets_config["data_type"]
        )

        # Instantiate the PyTorch model
        self.hstu_dense_model = HSTUDenseNet()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output_dtype = self.output_dtype
        seq_offsets_dtype = self.seq_offsets_config

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT
            # hidden_state = pb_utils.get_input_tensor_by_name(request, "HIDDEN_STATE")
            user_ids = pb_utils.get_input_tensor_by_name(request, "USER_IDS")
            # seq_len = pb_utils.get_input_tensor_by_name(request, "SEQ_LEN")
            # num_candidates = pb_utils.get_input_tensor_by_name(request, "NUM_CANDIDATES")

            output, num_cached_length, seqlen_offsets, num_candidates_offsets = self.hstu_dense_model(
                from_dlpack(hidden_state), from_dlpack(user_ids), from_dlpack(seq_len), from_dlpack(num_candidates))

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output = pb_utils.Tensor("OUTPUT", to_dlpack(output.astype(output_dtype)))
            num_cached_length = pb_utils.Tensor("NUM_CACHED_LEN", to_dlpack(num_cached_length.astype(seq_offsets_dtype)))
            seqlen_offsets = pb_utils.Tensor("SEQ_OFFSETS", to_dlpack(seqlen_offsets.astype(seq_offsets_dtype)))
            num_candidates_offsets = pb_utils.Tensor("NUM_CANDIDATES_OFFSETS", to_dlpack(num_candidates_offsets.astype(seq_offsets_dtype)))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output, num_cached_length, seqlen_offsets, num_candidates_offsets]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
