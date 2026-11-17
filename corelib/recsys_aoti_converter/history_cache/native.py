# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Serialized fixed-history adapter over the shared corelib owner and FlexKV.

The runtime synchronizes at transfer/lifetime boundaries and supports one
serialized owner thread per process.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading

import torch

from .precision import DTYPES, precision_name


@dataclass
class Request:
    ids: torch.Tensor
    lengths: torch.Tensor
    lookup: list[torch.Tensor]
    allocation: list[torch.Tensor]
    tasks: torch.Tensor
    table: torch.Tensor
    miss_rows: list[int]
    tables: list[torch.Tensor]


class NativeCache:
    _active = False
    def __init__(self, project: Path, reference: Path, *, capacity=16, length=128, heads=4, head_dim=64, dtype=torch.float32, banks=1, owner_loop=True, batch_lookup=True, max_batch=8, cpu_capacity=None, fast_metadata=True, resident_fastpath=None):
        if length <= 0 or length & (length-1):
            raise ValueError('FlexKV requires a power-of-two fixed history slot length')
        if head_dim not in (64,128,256,512) or not 1 <= heads <= 32:
            raise ValueError('Unsupported append-kernel head dimensions')
        if capacity < 1:
            raise ValueError('Cache capacity must be positive')
        if banks < 1:
            raise ValueError('At least one history bank is required')
        if not 1 <= max_batch <= 256:
            raise ValueError('Maximum batch must be in 1..256')
        self.precision = precision_name(dtype)
        self.dtype = DTYPES[self.precision]
        self.project, self.reference = Path(project), Path(reference)
        self.length, self.heads, self.head_dim, self.capacity = length, heads, head_dim, capacity
        self.banks = banks
        self.max_batch = max_batch
        self.cpu_capacity = max(64,capacity*4) if cpu_capacity is None else cpu_capacity
        if self.cpu_capacity < max_batch:
            raise ValueError('CPU cache must accommodate at least one maximum-sized publication batch')
        self.owner_loop = owner_loop
        # This adapter always uses complete fixed-length snapshot slots. Keep
        # its metadata path aligned with the resident-hit runtime specialization.
        self.batch_lookup = bool(batch_lookup)
        self.fast_metadata = bool(fast_metadata)
        if resident_fastpath is None:
            resident_fastpath = os.environ.get('HISTORY_RESIDENT_FASTPATH', '1') != '0'
        self.resident_fastpath = bool(resident_fastpath)
        self.lock = threading.RLock()
        self.directory = Path(tempfile.mkdtemp(prefix='history_flexkv_'))
        self.server = None
        self.opened = False
        self.failed = False
        self.owner_thread = None
        cfg = {
            'num_layers':banks,'num_kv_heads':heads,'head_size':head_dim,'dtype':str(self.dtype).removeprefix('torch.'),
            'tokens_per_page':length,'tokens_per_chunk':length,'num_primary_cache_pages':capacity,
            'num_buffer_pages':0,'max_batch_size':max_batch,'max_sequence_length':length,'device_idx':0,
            'fixed_history_slots':1,
            # history_single uses the shared CacheOwner directly. Keep the
            # standard-op facade off so one runtime has exactly one owner.
            'owner_loop':0,
            # The supported FlexKV revision retains child tasks; preserve their
            # handles until completion and explicitly reap them.
            'onboard_child_tasks':1,
            'batch_lookup':int(self.batch_lookup),
            'fast_metadata':int(self.fast_metadata),
            'server_recv_port':f'ipc://{self.directory}/server.sock',
            'gpu_register_port':f'ipc://{self.directory}/gpu.sock'}
        self.config = cfg
        path = self.directory/'runtime.yaml'
        path.write_text(''.join(f'{k}: {v}\n' for k,v in cfg.items()))
        os.environ['KVCACHE_MANAGER_CONFIG_FILE'] = str(path)
        os.environ['KVCACHE_MANAGER_OPS_LIB'] = str(self.project/'build/manager/kvcache_manager_ops.so')
        os.environ['FLEXKV_ENABLE_MPS'] = '0'
        os.environ['FLEXKV_SERVER_CLIENT_MODE'] = '1'
        sys.path.insert(0,str(self.reference/'corelib/recsys_kvcache_manager'))

    def open(self):
        if NativeCache._active:
            raise RuntimeError('Only one native cache runtime is supported per process')
        self.owner_thread = threading.get_ident()
        os.environ.setdefault('FLEXKV_LOG_LEVEL', 'WARNING')
        from flexkv.common.config import ModelConfig, CacheConfig
        from flexkv.server.server import KVServer
        model = ModelConfig(num_layers=self.banks,num_kv_heads=self.heads,head_size=self.head_dim,dtype=self.dtype)
        cache = CacheConfig(tokens_per_block=self.length,num_cpu_blocks=self.cpu_capacity,num_tmp_cpu_blocks=max(16,self.max_batch),
                            num_local_blocks=self.cpu_capacity,enable_ssd=False,enable_p2p_cpu=False,enable_p2p_ssd=False)
        self.server = KVServer.create_server(model_config=model,cache_config=cache,
                    gpu_register_port=self.config['gpu_register_port'],server_recv_port=self.config['server_recv_port'],
                    total_clients=1,inherit_env=True)
        import recsys_kvcache_manager
        recsys_kvcache_manager.load_kvcache_manager_ops(strict=True)
        # A model importing the full RecSys operators may have registered the
        # compatible append op already. Never register the same schema twice.
        append = getattr(torch.ops.paged_kvcache_ops, 'append_kvcache', None)
        if append is None:
            torch.ops.load_library(str(self.project/'build/append/recsys_paged_kvcache_ops.so'))
            append = torch.ops.paged_kvcache_ops.append_kvcache
        if not hasattr(append, 'mutate'):
            raise RuntimeError(
                'The registered RecSys append operators lack append_kvcache.mutate. '
                'Rebuild the matching updated native operators and restart the process; '
                'loading a second append library would duplicate existing schemas.')
        if self.max_batch != 8 or hasattr(torch.ops.history_single,'set_request_limit'):
            from . import single_ops
            single_ops.load(self.project)
            single_ops.set_max_batch_size(self.max_batch)
            torch.ops.history_single.set_request_limit(self.max_batch)
        from recsys_kvcache_manager import fake_kvcache_manager_ops
        fake_kvcache_manager_ops._RUNTIME_CONFIG = None
        fake = self.reference/'examples/commons/ops/cuda_ops/fake_paged_kvcache_ops.py'
        if 'history_cache._paged_fake' not in sys.modules:
            spec = importlib.util.spec_from_file_location('history_cache._paged_fake',fake)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules['history_cache._paged_fake'] = module
        if self.owner_loop:
            from . import single_ops
            single_ops.load(self.project)
            if hasattr(torch.ops.history_single, 'set_resident_fastpath'):
                torch.ops.history_single.set_resident_fastpath(self.resident_fastpath)
            else:
                self.resident_fastpath = False
            self.tables = torch.ops.history_single.owner_start(torch.empty(0,device='cuda'))
            if hasattr(torch.ops.history_single, 'resident_fastpath_enabled'):
                self.resident_fastpath = bool(torch.ops.history_single.resident_fastpath_enabled())
        else:
            self.resident_fastpath = False
            torch.ops.kvcache_manager_ops.init_kvcache(torch.empty(0,dtype=torch.int64))
        # Keep GPU storage alive until FlexKV transfer workers have exited,
        # including when the very first request fails before publishing slots.
        if not self.owner_loop:
            self.tables = torch.ops.kvcache_manager_ops.get_cache_tables(torch.empty(0,dtype=torch.int64))
        self.table = self.tables[0]  # compatibility accessor for one-bank callers
        self.opened=True
        NativeCache._active=True
        return self

    def assert_owner_thread(self):
        if self.owner_thread is not None and threading.get_ident() != self.owner_thread:
            raise RuntimeError('owner_thread_required: native KV state is thread-local; call on the opening thread or use a separate process instance')

    def prepare(self, ids, *, wait=True):
        self.assert_owner_thread()
        if self.owner_loop:
            raise RuntimeError('The owner loop uses the single-program prepare operation')
        ids=torch.as_tensor(ids,dtype=torch.int64,device='cpu')
        if ids.ndim!=1 or len(ids)==0 or len(ids)>min(self.capacity,self.max_batch):
            raise ValueError('Request exceeds the configured history batch limit')
        if len(set(ids.tolist())) != len(ids):
            raise ValueError('NativeCache.prepare expects deduplicated identities')
        lengths=torch.full_like(ids,self.length)
        op=torch.ops.kvcache_manager_ops
        found=op.lookup(ids,lengths,ids)
        if any(x not in (0,self.length) for x in found[1].tolist()):
            raise ValueError('Partial snapshot match is not supported by the fixed-slot adapter')
        misses=(found[1]==0).nonzero().flatten().tolist()
        alloc=op.allocate(ids,lengths,found[1],found[5])
        launch=op.onboard_launch(ids,lengths,found,alloc[0],alloc[2])
        tables=op.onboard_wait(launch[2],alloc[0]) if wait else op.get_cache_tables(ids)
        if alloc[0].numel()!=ids.numel():
            raise RuntimeError('Expected exactly one complete history slot per identity')
        return Request(ids,lengths,found,alloc,launch[2],tables[0],misses,tables)

    def fill(self, request, key, value):
        self.assert_owner_thread()
        n=len(request.miss_rows)
        expected=(n,self.heads,self.length*self.banks,self.head_dim)
        if tuple(key.shape)!=expected or tuple(value.shape)!=expected:
            raise ValueError(f'Expected missing K/V shape {expected}')
        if any(x.dtype != self.dtype or x.device != request.table.device for x in (key,value)):
            raise ValueError(f'K/V dtype and device must match the {self.dtype} GPU cache')
        if n==0:
            return request.table
        from .modules import FixedHistoryFill
        slots=request.allocation[0][torch.tensor(request.miss_rows,device='cuda')]
        return FixedHistoryFill(self.length,self.heads,self.head_dim,banks=self.banks,max_batch=self.max_batch)(key,value,slots,*request.tables)

    def read(self, request):
        self.assert_owner_thread()
        selected=[table.index_select(0,request.allocation[0].to(torch.int64)) for table in request.tables]
        return tuple(torch.cat([x[:,i].permute(0,2,1,3) for x in selected],dim=2).contiguous() for i in (0,1))

    def offload(self, request):
        self.assert_owner_thread()
        torch.cuda.synchronize()  # legacy native op does not enforce producer completion
        found,alloc=request.lookup,request.allocation
        return torch.ops.kvcache_manager_ops.offload_launch(request.ids,request.lengths,found[1],found[5],
                found[2],found[3],alloc[0],alloc[2],[torch.tensor([-1],dtype=torch.int64)],request.ids)

    def flush(self):
        self.assert_owner_thread()
        if self.owner_loop:
            return torch.ops.history_single.owner_flush(torch.empty(0,dtype=torch.int64))
        return torch.ops.kvcache_manager_ops.offload_flush(torch.empty(0,dtype=torch.int64))

    def owner_stats(self):
        self.assert_owner_thread()
        if not self.owner_loop:
            return None
        values = torch.ops.history_single.owner_stats(torch.empty(0,dtype=torch.int64)).tolist()
        return dict(zip(('queue_high_water', 'onboard_polls', 'offload_launches', 'offload_reaped',
                         'admission_polls', 'offload_pending', 'offload_queued', 'onboard_pending'), values))

    def evict(self, ids):
        self.assert_owner_thread()
        torch.cuda.synchronize()
        ids=torch.as_tensor(ids,dtype=torch.int64)
        if self.owner_loop:
            return torch.ops.history_single.owner_evict(ids)
        if torch.ops.kvcache_manager_ops.offload_pending(ids):
            self.flush()
        return torch.ops.kvcache_manager_ops.evict_kvcache(ids,True,ids)

    def close(self):
        self.assert_owner_thread()
        try:
            if self.opened:
                torch.cuda.synchronize()
                if self.owner_loop:
                    torch.ops.history_single.owner_stop(torch.empty(0,dtype=torch.int64))
                else:
                    if not self.failed:
                        self.flush()
                    torch.ops.kvcache_manager_ops.shutdown_runtime(torch.empty(0,dtype=torch.int64))
        finally:
            self.opened=False
            NativeCache._active=False
            if self.server is not None:
                self.server.shutdown()
                self.server=None
            self.table = None
            self.tables = None

    def __enter__(self):
        try:
            return self.open()
        except BaseException:
            self.close()
            raise

    def __exit__(self,*args):
        self.close()
