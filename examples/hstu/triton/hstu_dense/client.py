# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
# import triton_python_backend_utils as pb_utils

model_name = "hstu_dense"
shape = [5]

with httpclient.InferenceServerClient("localhost:8000") as client:
    # hidden_state = torch.rand([10, 1024], dtype=torch.bfloat16)
    # user_ids = torch.tensor([5], dtype=torch.int64)
    # seq_lens = torch.tensor([10], dtype=torch.int64)
    # num_candidates = torch.tensor([5], dtype=torch.int64)

    seq_lens = np.array([ 5, 3, 7, 2, 3], dtype=np.int64)
    inputs = [
        httpclient.InferInput(
            "SEQ_LEN", seq_lens.shape, np_to_triton_dtype(seq_lens.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(seq_lens)

    outputs = [
        httpclient.InferRequestedOutput("SEQ_OFFSETS"),
    ]

    response = client.infer(model_name, inputs, request_id=str(0), outputs=outputs)

    result = response.get_response()
    print(result)
    seqlen_offsets = response.as_numpy("SEQ_OFFSETS")

    print(
        "SEQ_LEN ({}) : SEQ_OFFSETS ({})".format(
            seq_lens, seqlen_offsets
        )
    )

    print("PASS: pytorch")
    sys.exit(0)
