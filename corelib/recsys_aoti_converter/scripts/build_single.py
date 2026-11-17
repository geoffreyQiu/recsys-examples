# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build fixed-slot graph adapters linked to the shared corelib cache owner."""
from pathlib import Path
import argparse
import os
import shlex
import torch
from torch.utils.cpp_extension import load

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--reference', type=Path,
               help='RecSys source tree containing corelib/recsys_kvcache_manager')
p.add_argument('--manager-library', type=Path,
               default=root/'build/manager/kvcache_manager_ops.so')
a=p.parse_args()
library=a.manager_library.resolve()
if not library.is_file():
    p.error(f'Build the shared KV manager first: {library}')

def owner_headers():
    candidates=[]
    if a.reference is not None:
        candidates.append(a.reference/'corelib/recsys_kvcache_manager/src/runtime')
    elif os.environ.get('RECSYS_REFERENCE'):
        candidates.append(Path(os.environ['RECSYS_REFERENCE'])/'corelib/recsys_kvcache_manager/src/runtime')
    else:
        # build_native.py/CMake records the exact manager source used.
        # Keep the two-command build tied to those same source headers.
        cache=library.parent/'CMakeCache.txt'
        if cache.is_file():
            for line in cache.read_text().splitlines():
                if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL='):
                    candidates.append(Path(line.split('=',1)[1])/'src/runtime')
        candidates.extend([
            root.parent/'recsys_kvcache_manager/src/runtime',
        ])
    for directory in candidates:
        if (directory/'cache_owner.h').is_file():
            return directory.resolve()
    p.error('The shared corelib cache_owner.h is required; pass --reference to the updated RecSys source tree')

headers=owner_headers()
major,minor=torch.cuda.get_device_capability()
os.environ.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')
out=root/'build/single'
out.mkdir(parents=True,exist_ok=True)
result=load(name='history_single_ops',sources=[str(root/'csrc/single_cache.cpp'),str(root/'csrc/single_cache_cuda.cu')],
             extra_include_paths=[str(headers)],
             extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],
             extra_ldflags=[f'-L{shlex.quote(str(library.parent))}', f'-l:{library.name}',
                            "-Wl,-rpath,'$$ORIGIN/../manager'"],
             build_directory=str(out),is_python_module=False,verbose=True)
print('SHARED_OWNER_LIBRARY',library,flush=True)
print('SINGLE_LIBRARY',result,flush=True)
