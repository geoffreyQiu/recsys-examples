# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Native fixed-slot kernels and real FlexKV transfers, before model compilation."""
import json
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from history_cache.native import NativeCache
from history_cache import single_ops


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    op=torch.ops.history_single
    results=[]
    with NativeCache(root,args.reference,dtype=torch.bfloat16,banks=1,length=32,heads=2,head_dim=64) as cache:
        single_ops.load(root)
        def run(ids,expected_misses,offload=False,expected_cpu=0):
            # Every identity has distinct, reproducible K/V; duplicates agree.
            x=torch.tensor(ids,device='cuda',dtype=torch.bfloat16)[:,None,None,None]
            key=(x+torch.arange(32,device='cuda',dtype=torch.bfloat16)[None,None,:,None]/32).expand(-1,2,-1,64)
            value=-key
            ticket,rows,slots=op.prepare(torch.tensor(ids),key,32,1)
            pk,pv,missing=op.pack([key,value],rows,slots,ticket)
            filled=op.fill(cache.table,missing,pk,pv,ticket,0)
            ready=op.onboard_wait(ticket,filled,key)
            actual=op.gather(cache.table,slots,ready,0)
            end=op.offload_launch(ticket,actual[0],torch.tensor([offload]))
            stats=op.retire(end).tolist()
            torch.testing.assert_close(actual[0],key,rtol=0,atol=0)
            torch.testing.assert_close(actual[1],value,rtol=0,atol=0)
            assert stats==[len(set(ids)),expected_misses,expected_cpu,1,1,1,1,int(offload)],stats
            results.append({'ids':ids,'stats':stats,'bit_exact':True})
            print('PASS',results[-1],flush=True)
        run([1,2,1,3],3)
        run([3,1,2,1],0)
        run([4,2,1,4],1)
        run([5],1)
        run([5],0)
        run([6,7,8,9,10,11,12,13],8,offload=True)
        cache.flush()
        cache.evict([6,7,8,9,10,11,12,13])
        run([6,7,8,9,10,11,12,13],0,expected_cpu=8)
        op.check_idle(torch.empty(0))
    path=root/'artifacts/single_program/native_results.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n')
    print('SINGLE_NATIVE_OK',flush=True)


if __name__=='__main__':main()
