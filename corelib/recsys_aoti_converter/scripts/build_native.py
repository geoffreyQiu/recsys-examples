# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build only the runtime and append libraries from the RecSys source tree."""
from pathlib import Path
import argparse
import subprocess
import os
import torch
from torch.utils.cpp_extension import load

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.source=a.source.resolve()
a.output=a.output.resolve()
major,minor=torch.cuda.get_device_capability()
os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')
a.output.mkdir(parents=True,exist_ok=True)
source=a.source/'corelib/recsys_kvcache_manager'
build=a.output/'manager'
subprocess.run(['cmake','-S',str(source),'-B',str(build),'-G','Ninja',
                f'-DCMAKE_CUDA_ARCHITECTURES={major}{minor}','-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path],check=True)
subprocess.run(['cmake','--build',str(build),'--parallel','4'],check=True)
append=a.source/'examples/commons/ops/cuda_ops/csrc'
(a.output/'append').mkdir(exist_ok=True)
library=load(name='recsys_paged_kvcache_ops',sources=[str(append/'paged_kvcache_ops_cuda.cpp'),str(append/'paged_kvcache_ops_kernel.cu')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3'],build_directory=str(a.output/'append'),
             is_python_module=False,verbose=True)
print('MANAGER_LIBRARY',build/'kvcache_manager_ops.so',flush=True)
print('APPEND_LIBRARY',library,flush=True)
