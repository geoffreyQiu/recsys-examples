# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE}
ARG FLEXKV_REV=0a732058d1c0b2c6755058c5827de76cb19e24b2
ARG CUDA_ARCH=9.0
ARG BUILD_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    MAX_JOBS=${BUILD_JOBS} \
    FLEXKV_ENABLE_METRICS=0 \
    FLEXKV_ENABLE_GDS=0 \
    FLEXKV_ENABLE_P2P=0 \
    FLEXKV_ENABLE_CFS=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake ninja-build pkg-config \
    libzmq3-dev cppzmq-dev liburing-dev libxxhash-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/flexkv-src
RUN git init && git remote add origin https://github.com/taco-project/FlexKV.git \
    && git fetch --depth 1 origin ${FLEXKV_REV} && git checkout --detach FETCH_HEAD \
    && git submodule update --init --depth 1 third_party/xxHash
RUN python3 -m pip install --no-cache-dir \
    pyzmq psutil nvtx pyyaml 'Cython>=3.0.10' 'expiring-dict==1.1.2' redis pytest-benchmark \
    && cmake -S . -B build -DFLEXKV_ENABLE_MONITORING=OFF \
    && cmake --build build --parallel ${BUILD_JOBS} \
    && FLEXKV_DEBUG=0 python3 -m pip install --no-build-isolation --no-deps .
WORKDIR /workspace/recsys-examples/corelib/recsys_aoti_converter
RUN python3 -c 'import torch, flexkv; from flexkv.server.server import KVServer; assert hasattr(KVServer, "create_server"); print(torch.__version__, flexkv.__file__)'
