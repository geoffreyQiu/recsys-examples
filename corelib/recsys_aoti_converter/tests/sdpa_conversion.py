# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Numerical mask/scale tests, conservative exclusions and compiled dynamic B."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.attention import replace_attention
from history_cache.precision import configure_math, compare, TOLERANCES
from history_cache.single_capture import capture
from history_cache.single import options


class Attention(torch.nn.Module):
    def __init__(self, mode='masked'):
        super().__init__()
        self.mode = mode

    def forward(self, q, k, v, valid):
        scores = torch.matmul(q, k.transpose(-1,-2)) / 8.0
        if self.mode != 'unmasked':
            bias = (1.0-valid.to(scores.dtype)[:,None,None,:]) * -1e9
            if self.mode == 'boolean_bias':
                bias = valid[:,None,None,:]
            scores = scores + bias
        probs = torch.softmax(scores, dim=-2 if self.mode=='wrong_axis' else -1)
        if self.mode == 'dropout':
            probs = torch.dropout(probs,0.1,True)
        out = torch.matmul(probs,v)
        return (out,probs) if self.mode=='extra_consumer' else out


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(921)
    torch.set_num_threads(4)
    result={'cases':[],'negative_cases':[]}
    for precision in ('fp32','bf16'):
        configure_math(precision,False)
        # Exercise both FP32 and BF16 attention with an explicit input dtype.
        inputs=(torch.randn(4,2,4,64,device='cuda')*.1,
                torch.randn(4,2,16,64,device='cuda')*.1,
                torch.randn(4,2,16,64,device='cuda')*.1,
                torch.ones(4,16,device='cuda',dtype=torch.bool))
        # Explicitly cast Q/K/V because these fixtures have no preceding
        # linear to establish the projected activation dtype.
        if precision=='bf16':
            inputs=tuple(x.bfloat16() if x.is_floating_point() else x for x in inputs)
        for mode in ('masked','unmasked'):
            ep,_=capture(Attention(mode),inputs,precision)
            before=ep.module()
            after=ep.module()
            report=replace_attention(after)
            assert report['matched']==1,report
            dim=torch.export.Dim('batch',min=1,max=8)
            converted=torch.export.export(after,inputs,dynamic_shapes=tuple({0:dim} for _ in inputs),strict=False)
            package=a.output/f'{precision}_{mode}.pt2'
            torch._inductor.aoti_compile_and_package(converted,package_path=str(package),inductor_configs=options(precision))
            compiled=torch._inductor.aoti_load_package(str(package))
            for batch in (1,2,4,8):
                current=tuple(x.index_select(0,torch.arange(batch,device='cuda')%4) for x in inputs)
                for mask in ('all_valid','partial','all_masked'):
                    valid=current[-1].clone()
                    if mask=='partial':valid[:,::2]=False
                    if mask=='all_masked':valid.zero_()
                    args=(*current[:-1],valid)
                    original=before(*args)
                    actual=compiled(*args)
                    if isinstance(actual,(tuple,list)):actual=actual[0]
                    metrics=compare(actual,original,tolerance=TOLERANCES[precision])
                    assert metrics['numeric_within_tolerance'] and metrics['finite'],(precision,mode,batch,mask,metrics)
                    if mask=='all_masked' and mode=='masked':
                        # Finite-bias masking keeps the small-score fixture's
                        # uniform distribution, unlike an empty boolean mask.
                        expected=args[2].float().mean(-2,keepdim=True).expand_as(actual).to(actual.dtype)
                        torch.testing.assert_close(actual,expected,**TOLERANCES[precision])
                        assert bool(actual.abs().max()>0)
                    result['cases'].append({'precision':precision,'mode':mode,'batch':batch,'mask':mask,**metrics})
        for mode in ('wrong_axis','extra_consumer','dropout','boolean_bias'):
            ep,_=capture(Attention(mode),inputs,precision)
            report=replace_attention(ep.module())
            assert report['matched']==0,(mode,report)
            result['negative_cases'].append({'precision':precision,'mode':mode,'skipped':report['skipped']})
    result['status']='passed'
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SDPA_REGRESSION_PASS',len(result['cases']),len(result['negative_cases']),flush=True)


if __name__=='__main__':main()
