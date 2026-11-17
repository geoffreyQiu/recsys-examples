# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Audit the emitted AOTI wrapper, not just the pre-compilation FX graph."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile


def math_kernel_counts(cpp):
    return {op:len(re.findall(r'aoti_torch_cuda_'+op+r'\(',cpp))
        for op in ('bmm_out','mm_out','addmm_out','_scaled_dot_product_efficient_attention',
                   '_scaled_dot_product_cudnn_attention','_scaled_dot_product_flash_attention')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    with zipfile.ZipFile(a.bundle/'model_aoti.pt2') as archive:
        cpp=archive.read(next(n for n in archive.namelist() if n.endswith('.wrapper.cpp'))).decode()
        metadata=json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('.wrapper.json'))))
    out=a.bundle/'compiled_audit';out.mkdir(exist_ok=True)
    (out/'wrapper.cpp').write_text(cpp)
    (out/'extern_nodes.json').write_text(json.dumps(metadata,indent=2)+'\n')
    calls=[]
    for lineno,line in enumerate(cpp.splitlines(),1):
        match=re.search(r'aoti_torch_proxy_executor_call_function\(proxy_executor, (\d+),',line)
        if match:
            index=int(match[1]);node=metadata['nodes'][index]['node']
            if node['target'].startswith('history_single::'):
                calls.append({'line':lineno,'proxy_index':index,'op':node['target'],
                              'inputs':node['inputs'],'outputs':node['outputs'],'code':line.strip()})
    names=[c['op'].split('::')[1] for c in calls]
    expected=['prepare','pack']+['fill','onboard_wait','gather']*manifest['bank_count']+['offload_launch']
    # Independent banks can be scheduled before another bank's gather. Check
    # the real token/pool dependencies, while retaining the exact linear-order
    # comparison as separate evidence instead of concealing rescheduling.
    def input_arg(call,name):
        return next(x['arg'] for x in call['inputs'] if x['name']==name)
    def output_tensor(call):
        return call['outputs'][0]['as_tensor']['name']
    counts=dict(Counter(names))
    checks={'counts':counts==dict(Counter(expected)),
            'prepare_then_pack':names[:2]==['prepare','pack'],
            'offload_last':bool(names) and names[-1]=='offload_launch'}
    bank_checks=[]
    for bank in range(manifest['bank_count']):
        fills=[c for c in calls if c['op']=='history_single::fill' and input_arg(c,'bank')['as_int']==bank]
        gathers=[c for c in calls if c['op']=='history_single::gather' and input_arg(c,'bank')['as_int']==bank]
        entry={'bank':bank,'one_fill_and_gather':len(fills)==len(gathers)==1}
        if entry['one_fill_and_gather']:
            fill,gather=fills[0],gathers[0]
            waits=[c for c in calls if c['op']=='history_single::onboard_wait'
                   and input_arg(c,'filled')['as_tensor']['name']==output_tensor(fill)]
            entry['one_matching_wait']=len(waits)==1
            entry['same_pool']=input_arg(fill,'pool')==input_arg(gather,'pool')
            if len(waits)==1:
                wait=waits[0]
                entry['fill_before_wait_before_gather']=fill['line']<wait['line']<gather['line']
                entry['gather_uses_matching_ready']=input_arg(gather,'ready')['as_tensor']['name']==output_tensor(wait)
                entry['same_request_ticket']=input_arg(fill,'ticket')==input_arg(wait,'ticket')==input_arg(calls[-1],'ticket')
        entry['passed']=all(value for key,value in entry.items() if key!='bank')
        bank_checks.append(entry)
    checks['all_bank_dependencies']=all(b['passed'] for b in bank_checks)
    passed=all(checks.values())
    report={'calls':calls,'actual_order':names,'expected_order':expected,
            'order_matches':names==expected,'counts':dict(Counter(names)),
            'dependency_order_passed':passed,'checks':checks,'bank_checks':bank_checks,
            'scope':'emitted C++ host launch order and each bank fill/wait/gather token/pool dependencies; strict interleaving reported separately; CUDA/IPC completion tested at runtime'}
    report['math_policy']=manifest.get('math_policy',{'tf32':False,'source':'legacy TF32-disabled bundle'})
    report['math_kernel_calls']=math_kernel_counts(cpp)
    if manifest.get('sdpa'):
        report['uncached_controls']={}
        for name in ('uncached_aoti.pt2','uncached_sdpa_aoti.pt2'):
            package=a.bundle/name
            with zipfile.ZipFile(package) as archive:
                control_cpp=archive.read(next(n for n in archive.namelist() if n.endswith('.wrapper.cpp'))).decode()
                meta_name=next((n for n in archive.namelist() if n.endswith('.wrapper.json')),None)
                # Packages with no dispatcher proxy calls omit this file.
                if meta_name is None:
                    assert not re.search(r'aoti_torch_proxy_executor_call_function\(',control_cpp)
                    control_meta={'nodes':[]}
                else:
                    control_meta=json.loads(archive.read(meta_name))
            cache_calls=[n['node']['target'] for n in control_meta['nodes']
                         if n['node']['target'].startswith('history_single::')]
            report['uncached_controls'][name]={
                'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
                'math_kernel_calls':math_kernel_counts(control_cpp),
                'cache_calls':cache_calls}
            assert not cache_calls,(name,cache_calls)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert passed,report
    print('COMPILED_DEPENDENCIES_PASS','strict_interleaving=',names==expected,names)


if __name__=='__main__':main()
