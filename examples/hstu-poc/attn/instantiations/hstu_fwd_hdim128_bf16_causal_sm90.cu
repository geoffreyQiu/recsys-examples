
/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_kernels.py

#include "hstu_fwd_launch_template.h"

template void run_hstu_fwd_<90, cutlass::bfloat16_t, 128, false, false, true, false, false, false, 0>
                           (Hstu_fwd_params& params, cudaStream_t stream);

    