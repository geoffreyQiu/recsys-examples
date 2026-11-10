"""Run inside the built image with --gpus; fails on missing imports/kernels."""
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path

import torch

assert os.environ.get('HSTU_INFERENCE_ONLY') == '1'
assert torch.cuda.is_available(), 'Container has no usable GPU; check driver and NVIDIA Container Toolkit'
cap = torch.cuda.get_device_capability()
free_bytes, total_bytes = torch.cuda.mem_get_info()
assert cap == (12, 0), f'Unexpected GPU capability: {cap}'
for name in ['torchrec', 'dynamicemb', 'flexkv.c_ext', 'hstu', 'hstu_cuda_ops',
             'paged_kvcache_ops', 'recsys_kvcache_manager']:
    importlib.import_module(name)
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
y = x @ x
assert torch.isfinite(y).all()
torch.cuda.synchronize()
print(json.dumps({
    'gpu': torch.cuda.get_device_name(), 'capability': cap,
    'free_gpu_gib': round(free_bytes / 1024**3, 2), 'total_gpu_gib': round(total_bytes / 1024**3, 2),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'packages': {p: md.version(p) for p in ['fbgemm-gpu', 'torchrec', 'dynamicemb', 'flexkv']},
}, indent=2))
# Import exactly the benchmark's entry point without launching the model.
import runpy
runpy.run_path('inference/benchmark/inference_benchmark_flexkv.py', run_name='preflight_import')
print('PASS: GPU operation and benchmark imports')
