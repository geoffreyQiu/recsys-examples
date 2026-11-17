# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Capture and convert a declared model through the one-program API."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils import _pytree as pytree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from history_cache.package import load_package
from history_cache.convert import CacheSpec,partition_history
from history_cache.single_capture import capture
from history_cache.single import convert,options
from history_cache.native import NativeCache
from history_cache.precision import configure_math,check_math_policy
from history_cache.attention import replace_attention
from history_cache.review import source_catalog
from history_cache.dataflow_precision import validate_autocast,audit_dataflow


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--model-spec',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[3],
                   help='RecSys source tree containing corelib/recsys_kvcache_manager and examples/commons/ops/cuda_ops')
    p.add_argument('--precision',choices=('fp32','bf16'),default='bf16')
    p.add_argument('--autocast',action=argparse.BooleanOptionalAction,default=False,
                   help='Opt into BF16 mixed-precision autocast; default converts weights, floating inputs and FP32 graph dtype requests to BF16')
    p.add_argument('--tf32',action=argparse.BooleanOptionalAction,default=False,
                   help='Allow TF32 for FP32 export/compilation (default: disabled)')
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--owner-loop',action=argparse.BooleanOptionalAction,default=True,
                   help='Use the shared corelib cache owner (default); --no-owner-loop selects the legacy direct runtime')
    p.add_argument('--max-batch',type=int,default=8,help='Maximum batch size to export/compile (2..256)')
    p.add_argument('--capacity',type=int,default=None,help='GPU history slots; default max(16, 2*max-batch)')
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--reuse-capture',action='store_true')
    p.add_argument('--sdpa',action=argparse.BooleanOptionalAction,default=False,
                   help='Replace matched attention with SDPA after history matching; also build an uncached SDPA control')
    p.add_argument('--fold-attention',action='store_true',
                   help='Group independent candidate queries when SDPA K/V and bias are proven repeated broadcasts')
    p.add_argument('--review-diff',action='store_true',
                   help='Generate annotated original/converted FX diffs grouped by cache bank')
    a=p.parse_args()
    if a.fold_attention and not a.sdpa:p.error('--fold-attention requires --sdpa')
    if not 2<=a.max_batch<=256:p.error('--max-batch must be in 2..256')
    if a.capacity is None:a.capacity=max(16,2*a.max_batch)
    if a.capacity<a.max_batch:p.error('--capacity must cover one maximum-sized batch')
    validate_autocast(a.precision,a.autocast)
    math_policy=configure_math(a.precision,a.tf32)
    capture_policy={'precision':a.precision,'autocast':a.autocast,'preserve_explicit_fp64':True,**math_policy}
    if a.max_batch!=8:capture_policy['max_batch']=a.max_batch
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'source_model.json').write_text(a.model_spec.read_text())
    (a.output/'source.json').write_text(json.dumps({'package':str(a.package)})+'\n')
    print('MATH_POLICY',math_policy,flush=True)
    root=Path(__file__).resolve().parents[1]
    spec=CacheSpec(**json.loads(a.spec.read_text()))
    if a.reuse_capture:
        policy_path=a.output/'capture_policy.json'
        if not policy_path.exists() or json.loads(policy_path.read_text())!=capture_policy:
            raise ValueError('Reuse requires a capture with the same recorded precision/autocast/TF32 policy; recapture otherwise')
        ep=torch.export.load(a.output/'original_export.pt2')
        tree=pytree.treespec_loads(json.loads((a.output/'input_tree.json').read_text())['tree'])
    else:
        model,inputs,_,_=load_package(a.package,a.model_spec,min(4,a.max_batch))
        if a.review_diff:
            (a.output/'source_modules.json').write_text(json.dumps(source_catalog(model),indent=2)+'\n')
        ep,tree=capture(model,(inputs,),a.precision,max_batch=a.max_batch,autocast=a.autocast)
        torch.export.save(ep,a.output/'original_export.pt2')
        (a.output/'original_graph.txt').write_text(str(ep.graph))
        (a.output/'input_tree.json').write_text(json.dumps({'tree':pytree.treespec_dumps(tree)})+'\n')
        (a.output/'capture_policy.json').write_text(json.dumps(capture_policy,indent=2)+'\n')
    pure_bf16=a.precision=='bf16' and not a.autocast
    audits={'original':audit_dataflow(ep,require_bf16=pure_bf16)}
    (a.output/'dtype_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print('DYNAMIC_ORIGINAL_EXPORT_OK',ep.range_constraints,flush=True)
    check_math_policy(a.tf32)
    part=partition_history(ep,spec)
    print('BANKS',[b['prefix_length'] for b in part.report['banks']],flush=True)
    if a.capture_only:return
    with NativeCache(root,a.reference,banks=part.banks,length=spec.length,
                     heads=spec.heads,head_dim=spec.head_dim,
                     capacity=a.capacity,max_batch=a.max_batch,owner_loop=a.owner_loop,
                     dtype=getattr(torch,'bfloat16' if a.precision=='bf16' else 'float32')) as cache:
        manifest=convert(ep,spec,a.output,cache=cache,input_tree=tree,precision=a.precision,tf32=a.tf32,
                         autocast=a.autocast,compile=not a.export_only,sdpa=a.sdpa,review_diff=a.review_diff,
                         fold_attention=a.fold_attention)
    audits['cached']=json.loads((a.output/'converted_dtype_audit.json').read_text())
    if a.sdpa:
        control=ep.module()
        attention_report=replace_attention(control)
        if attention_report['matched']!=manifest['attention_replacements']:
            raise ValueError('Cached/uncached SDPA controls matched different attention counts')
        if a.fold_attention:
            from history_cache.attention_layout import fold_broadcast_attention
            attention_report['layout']=fold_broadcast_attention(control)
        (a.output/'uncached_sdpa_before_export.py').write_text(control.code)
        (a.output/'uncached_sdpa_report.json').write_text(json.dumps(attention_report,indent=2)+'\n')
        dim=torch.export.Dim('batch',min=1,max=a.max_batch)
        flat=tuple(ep.example_inputs[0])
        with torch.no_grad():
            sdpa_ep=torch.export.export(control,flat,dynamic_shapes=tuple({0:dim} for _ in flat),strict=False)
        torch.export.save(sdpa_ep,a.output/'uncached_sdpa_export.pt2')
        audits['uncached_sdpa']=audit_dataflow(sdpa_ep,require_bf16=pure_bf16)
        if not a.export_only:
            print('COMPILE_UNCACHED_SDPA',attention_report['matched'],flush=True)
            torch._inductor.aoti_compile_and_package(sdpa_ep,package_path=str(a.output/'uncached_sdpa_aoti.pt2'),inductor_configs=options(a.precision))
            print('UNCACHED_SDPA_OK',flush=True)
    (a.output/'dtype_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    if not a.export_only:
        print('COMPILE_UNCACHED_CONTROL',flush=True)
        torch._inductor.aoti_compile_and_package(ep,package_path=str(a.output/'uncached_aoti.pt2'),inductor_configs=options(a.precision))
        check_math_policy(a.tf32)
        print('UNCACHED_CONTROL_OK',flush=True)


if __name__=='__main__':main()
