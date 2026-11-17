# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Original-model capture with a flat, dynamic-batch input ABI."""
import torch
import torch.fx.experimental._config as symbolic_config
from torch.utils import _pytree as pytree
from .precision import precision_name
from .dataflow_precision import validate_autocast, lower_bf16, audit_dataflow


def export_dynamic(module, args, shape_factory):
    """Handle the pinned build's B!=1 layout guard without changing arithmetic.

    The fallback captures B2..max. Its compiled B1 execution must be validated
    separately; callers enforce the public input ABI and disable the optional
    generated validator only for this known layout specialization issue.
    """
    try:
        return torch.export.export(module,args,dynamic_shapes=shape_factory(1),strict=False)
    except torch._dynamo.exc.UserError as error:
        message=str(error)
        if 'Constraints violated (batch)' not in message or 'size()[0] != 1' not in message:
            raise
        print('EXPORT_LAYOUT_GUARD_FALLBACK: capture B2..max; compiled B1 validation required',flush=True)
        return torch.export.export(module,args,dynamic_shapes=shape_factory(2),strict=False)


class FlatPrecisionModel(torch.nn.Module):
    def __init__(self,model,tree,autocast=False):
        super().__init__()
        self.model=model
        self.tree=tree
        self.autocast=autocast

    def forward(self,*flat):
        args,kwargs=pytree.tree_unflatten(flat,self.tree)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.autocast):
            return self.model(*args,**kwargs)


def capture(model,args,precision='bf16',max_batch=8,*,autocast=False):
    precision=precision_name(precision)
    validate_autocast(precision,autocast)
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is not supported on the selected GPU')
    symbolic_config.backed_size_oblivious=True
    flat,tree=pytree.tree_flatten((args,{}))
    if not all(isinstance(x,torch.Tensor) and x.is_cuda and x.ndim for x in flat):
        raise ValueError('Expected batched CUDA model inputs')
    if len({x.shape[0] for x in flat})!=1 or flat[0].shape[0]<2:
        raise ValueError('Capture example must use a common batch >=2')
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        ep=export_dynamic(FlatPrecisionModel(model.eval(),tree,autocast),tuple(flat),
            lambda minimum:{'flat':tuple({0:torch.export.Dim('batch',min=minimum,max=max_batch)} for _ in flat)})
        # Even disabled autocast is captured as a higher-order subgraph.
        # Inline it for both precisions so structural K/V matching sees ATen
        # projections; the empty table retains linear ops and FP32 dtypes.
        ep=ep.run_decompositions({})
        if precision=='bf16' and not autocast:
            lowered,examples=lower_bf16(ep)
            ep=export_dynamic(lowered,tuple(examples),
                lambda minimum:tuple({0:torch.export.Dim('batch',min=minimum,max=max_batch)} for _ in examples))
            audit_dataflow(ep,require_bf16=True)
    return ep,tree
