# Build from repository root. Deployment default: RTX PRO 5500 (SM120).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG CUDA_ARCH=12.0
ARG DEMB_ARCHS=120
ARG BUILD_JOBS=4
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} DEMB_ARCHS=${DEMB_ARCHS} MAX_JOBS=${BUILD_JOBS} NVCC_THREADS=1
ENV HSTU_INFERENCE_ONLY=1
WORKDIR /workspace/recsys-examples
RUN mkdir -p /opt/build-driver && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/build-driver/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/build-driver:${LD_LIBRARY_PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git pkg-config liburing-dev libxxhash-dev libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /opt/build-driver/libnvidia-ml.so.1
# Match upstream's ABI-coupled FBGEMM/TorchRec pair; preserve NGC torch.
RUN python -m pip install --no-cache-dir --no-deps \
    'https://download.pytorch.org/whl/cu130/fbgemm_gpu-1.8.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=b5a98aad7c662c84e04a005d0656a7da99da15bd70c7aa1cb04bfc1e868a0b48' \
    && git clone --depth 1 --branch v1.8.0-rc1 https://github.com/meta-pytorch/torchrec.git /opt/torchrec \
    && test "$(git -C /opt/torchrec rev-parse HEAD)" = 0ea0ec09512930dbcbf6d24cb65132a77835bce8 \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/torchrec
RUN python -m pip install --no-cache-dir tensordict orjson pyre-extensions iopath \
    gin-config ordered-set expiring-dict==1.1.2 redis pyzmq pyyaml nvtx psutil \
    && python -c 'import torchrec; print(torchrec.__version__)'
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
COPY third_party/FlexKV /opt/FlexKV
ENV FLEXKV_DEBUG=0 FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_GDS=0 FLEXKV_ENABLE_P2P=0
RUN cmake -S /opt/FlexKV -B /opt/FlexKV/build -DFLEXKV_ENABLE_MONITORING=OFF \
    && cmake --build /opt/FlexKV/build --target xxhash -j${BUILD_JOBS} \
    && cd /opt/FlexKV && python -m pip install --no-build-isolation --no-deps .
COPY third_party/FBGEMM /opt/FBGEMM
# BF16, head_dim=256, forward only. Use upstream's SM80-family BF16
# implementation compiled to SM120a on RTX; disable separate FP8 SM120 path.
ENV HSTU_ARCH_LIST=${CUDA_ARCH} HSTU_DISABLE_120=TRUE HSTU_DISABLE_86OR89=FALSE \
    HSTU_DISABLE_BACKWARD=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_ARBITRARY=TRUE \
    HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DRAB=TRUE HSTU_DISABLE_FP16=TRUE \
    HSTU_DISABLE_FP8=TRUE HSTU_DISABLE_HDIM32=TRUE HSTU_DISABLE_HDIM64=TRUE HSTU_DISABLE_HDIM128=TRUE
RUN cd /opt/FBGEMM/fbgemm_gpu/experimental/hstu \
    && python -m pip install --no-build-isolation --no-deps .
# Required by paged_kvcache_ops, even with cache compression disabled.
RUN mkdir -p /workspace/deps && cd /workspace/deps \
    && wget -q https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz \
    && tar -xf nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz \
    && mv nvcomp-linux-x86_64-5.1.0.21_cuda12-archive nvcomp \
    && rm *.tar.xz
# Link-only CUDA stubs permit docker build without GPU access. Never add
# the stubs to runtime LD_LIBRARY_PATH; the runtime injects the real driver.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
COPY corelib/dynamicemb /workspace/recsys-examples/corelib/dynamicemb
RUN cd corelib/dynamicemb && python -m pip install --no-build-isolation --no-deps .
COPY examples/commons /workspace/recsys-examples/examples/commons
RUN cd examples/commons && BUILD_EXT_ONLY=hstu_cuda_ops,paged_kvcache_ops \
    python -m pip install --no-build-isolation --no-deps .
# The KV runtime client includes zmq.hpp (cppzmq), not just zmq.h.
# Keep this late so existing expensive extension layers remain cached.
RUN apt-get update && apt-get install -y --no-install-recommends cppzmq-dev \
    && test -f /usr/include/zmq.hpp \
    && rm -rf /var/lib/apt/lists/*
COPY corelib/recsys_kvcache_manager /workspace/recsys-examples/corelib/recsys_kvcache_manager
RUN cd corelib/recsys_kvcache_manager && python -m pip install --no-build-isolation --no-deps .
COPY examples/hstu /workspace/recsys-examples/examples/hstu
COPY docker/flexkv-minimal /workspace/recsys-examples/docker/flexkv-minimal
ENV PYTHONPATH=/workspace/recsys-examples/examples:/workspace/recsys-examples/examples/hstu
WORKDIR /workspace/recsys-examples/examples/hstu
RUN python inference/benchmark/inference_benchmark_flexkv.py --help
RUN rm -rf /opt/build-driver
CMD ["bash"]
