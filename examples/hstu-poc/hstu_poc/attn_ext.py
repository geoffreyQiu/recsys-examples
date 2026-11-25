"""JIT build of the copied FBGEMM Hopper HSTU forward kernel with the P4 wait hooks (attn/hstu_hopper, op namespace
torch.ops.hpoc_attn). Same flags as FBGEMM's setup.py for the reduced build, plus hidden visibility so its
symbols never bind to FBGEMM's identically named ones loaded in the same process."""
import glob
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATTN = os.path.join(ROOT, "attn")
CUTLASS = os.environ.get("HPOC_CUTLASS", "/lustre/fsw/coreai_devtech_all/junyiq/hstu-kv-rdma/repos/cutlass")
BUILD = os.environ.get("HPOC_ATTN_BUILD_DIR", os.path.join(ROOT, "build", "attn"))
FEATURES = ["-DHSTU_DISABLE_BACKWARD", "-DHSTU_DISABLE_DETERMINISTIC", "-DHSTU_DISABLE_LOCAL", "-DHSTU_DISABLE_CONTEXT",
            "-DHSTU_DISABLE_ARBITRARY", "-DHSTU_ARBITRARY_NFUNC=0", "-DHSTU_DISABLE_RAB", "-DHSTU_DISABLE_DRAB",
            "-DHSTU_DISABLE_FP16", "-DHSTU_DISABLE_FP8", "-DHSTU_DISABLE_HDIM32", "-DHSTU_DISABLE_HDIM64",
            "-DHSTU_DISABLE_86OR89", "-DHSTU_DISABLE_120"]
_mod = None


def attn_ext(verbose=False):
    global _mod
    if _mod is None:
        os.makedirs(BUILD, exist_ok=True)
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        srcs = [os.path.join(ATTN, "hstu_hopper", "hstu_ops_gpu.cpp")] + sorted(glob.glob(os.path.join(ATTN, "instantiations", "hstu_fwd_*.cu")))
        nvcc = ["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "--use_fast_math", "-DNDEBUG", "-lineinfo", "--threads", "4",
                "-gencode", "arch=compute_90a,code=sm_90a", "-Xcompiler", "-fvisibility=hidden"] + FEATURES
        _mod = load(
            name="hpoc_attn",
            sources=srcs,
            extra_include_paths=[os.path.join(CUTLASS, "include"), os.path.join(ATTN, "hstu_hopper"),
                                 os.path.join(cuda_home, "include", "cccl")],
            extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"] + FEATURES,
            extra_cuda_cflags=nvcc,
            extra_ldflags=["-lcuda"],
            build_directory=BUILD,
            is_python_module=False,
            verbose=verbose,
        )
    return torch.ops.hpoc_attn


def hstu_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scaling, num_targets, alpha, kv_ready=0, kv_seq=0, mode=0, pdl=False):
    """FBGEMM's hstu_varlen_fwd_90 (causal + targets), with the P4 wait settings applied to this launch only."""
    ops = attn_ext()
    ops.set_kv_wait(kv_ready, kv_seq, mode, pdl)
    try:
        out, _ = ops.hstu_varlen_fwd_90(q, k, v, cu_q, cu_k, None, None, max_q, max_k, scaling, None, num_targets, 1,
                                        -1, 0, alpha, None, None, -1, 0)
    finally:
        ops.set_kv_wait(0, 0, 0, False)
    return out
