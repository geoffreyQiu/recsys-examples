"""JIT build of the POC's C++/CUDA extension (transports + helpers). Build once per node before multi-rank runs:
    python -c "from hstu_poc.ext import ext; ext(verbose=True)"
"""
import os

from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSRC = os.path.join(ROOT, "csrc")
NCCL_HOME = os.environ.get("HPOC_NCCL_HOME", "/lustre/fsw/coreai_devtech_all/junyiq/src/nccl/build")
BUILD = os.environ.get("HPOC_BUILD_DIR", os.path.join(ROOT, "build", "ext"))
SOURCES = ["bindings.cpp", "gin_kv.cu", "kvutil.cu", "verbs_kv.cpp"]
_mod = None


def ext(verbose=False):
    global _mod
    if _mod is None:
        os.makedirs(BUILD, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _mod = load(
            name="hpoc_ext",
            sources=[os.path.join(CSRC, s) for s in SOURCES],
            extra_include_paths=[os.path.join(NCCL_HOME, "include"), os.path.join(cuda_home, "include", "cccl")],
            extra_cflags=["-O3", "-fopenmp", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-Xcompiler", "-fopenmp"],
            extra_ldflags=[f"-L{NCCL_HOME}/lib", "-lnccl", "-lcuda", "-libverbs", "-lgomp",
                           f"-Wl,-rpath,{NCCL_HOME}/lib"],
            build_directory=BUILD,
            verbose=verbose,
        )
    return _mod
