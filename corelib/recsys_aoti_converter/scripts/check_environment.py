# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Check the dependencies used by the standalone converter and native runtime."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import torch


def main():
    for name in ('numpy', 'flexkv', 'zmq', 'yaml', 'nvtx'):
        importlib.import_module(name)
    from flexkv.server.server import KVServer
    assert hasattr(KVServer, 'create_server'), 'FlexKV server/client API is required'
    assert hasattr(torch._inductor, 'aoti_compile_and_package')
    assert hasattr(torch.library, '_register_effectful_op')
    assert torch.cuda.is_available(), 'Expose the assigned CUDA GPU to this process'
    for name in ('cmake', 'ninja', 'nvcc', 'c++', 'pkg-config'):
        assert shutil.which(name), f'Missing build tool: {name}'
    subprocess.run(['pkg-config', '--exists', 'libzmq'], check=True)
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(),
                      'capability': torch.cuda.get_device_capability(),
                      'flexkv': importlib.metadata.version('flexkv'),
                      'flexkv_source': str(Path(importlib.import_module('flexkv').__file__).resolve()),
                      'dependency_check': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
