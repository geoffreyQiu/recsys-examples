# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fake metadata and effect contracts for the native common operators."""
from pathlib import Path
import torch

_LOADED=False
_HANDLES=[]
_MAX_BATCH=8


def set_max_batch_size(value):
    global _MAX_BATCH
    if not 1 <= value <= 256:
        raise ValueError('Supported export batch maximum is 1..256')
    _MAX_BATCH=value


def load(project):
    global _LOADED
    if _LOADED:return
    torch.ops.load_library(str(Path(project)/'build/single/history_single_ops.so'))
    @torch.library.register_fake('history_single::prepare')
    def prepare(ids,anchor,length,banks,context=None):
        cpu=ids.new_empty((1,))
        rows=ids.new_empty(ids.shape)
        slots=torch.empty((ids.shape[0],),dtype=torch.int32,device=anchor.device)
        return cpu,rows,slots
    @torch.library.register_fake('history_single::pack')
    def pack(inputs,rows,slots,ticket):
        m=torch.library.get_ctx().new_dynamic_size(min=0,max=_MAX_BATCH)
        return [x.new_empty((m,*x.shape[1:])) for x in inputs]+[slots.new_empty((m,))]
    @torch.library.register_fake('history_single::fill')
    def fill(pool,slots,key,value,ticket,bank):return ticket.new_empty(ticket.shape)
    @torch.library.register_fake('history_single::onboard_wait')
    def wait(ticket,filled,ordering):return ticket.new_empty(ticket.shape)
    @torch.library.register_fake('history_single::gather')
    def gather(pool,slots,ready,bank):
        shape=(slots.shape[0],pool.shape[3],pool.shape[2],pool.shape[4])
        return pool.new_empty(shape),pool.new_empty(shape)
    @torch.library.register_fake('history_single::offload_launch')
    def offload(ticket,ordering,enabled):return ticket.new_empty(ticket.shape)
    for name in ('prepare','onboard_wait','offload_launch'):
        _HANDLES.append(torch.library._register_effectful_op(getattr(torch.ops.history_single,name).default,torch.library.EffectType.ORDERED))
    _LOADED=True
