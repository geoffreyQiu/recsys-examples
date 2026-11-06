// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Schemas of INFERENCE_EMB operators with CPU kernels. They live in host-only
// code so a CPU-only build registers them without compiling any CUDA source.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m) {
  m.def("expand_table_ids(Tensor offsets, Tensor indices, Tensor? table_offsets_in_feature=None, int num_tables=0, int local_batch_size=1) -> Tensor");
  m.def("embedding_collection_index(Tensor marker, int marker_value, "
        "Tensor feature_ids, Tensor table_ids) -> Tensor");
}
