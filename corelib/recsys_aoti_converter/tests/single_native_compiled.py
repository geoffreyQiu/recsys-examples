# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""AOTI gate for real native effects, dynamic packing, and registered pool writes."""
import json
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('AOTI_RUNTIME_CHECK_INPUTS','0')
import torch
import torch.fx.experimental._config as symbolic_config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from history_cache.native import NativeCache
from history_cache import single_ops


class Model(torch.nn.Module):
    def forward(self,ids,x,pool,offload):
        op=torch.ops.history_single
        ticket,rows,slots=op.prepare(ids,x,32,1)
        packed,missing=op.pack([x],rows,slots,ticket)
        key=packed[:,None,:,None].expand(-1,2,-1,64).contiguous()
        filled=op.fill(pool,missing,key,-key,ticket,0)
        ready=op.onboard_wait(ticket,filled,x)
        key,value=op.gather(pool,slots,ready,0)
        output=key.float().sum((1,2,3))-value.float().sum((1,2,3))
        return output,op.offload_launch(ticket,output,offload)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts/single_program/native_compiled'
    out.mkdir(parents=True,exist_ok=True)
    symbolic_config.backed_size_oblivious=True
    with NativeCache(root,args.reference,dtype=torch.bfloat16,length=32,heads=2,head_dim=64) as cache:
        single_ops.load(root)
        dim=torch.export.Dim('batch',min=1,max=8)
        ep=torch.export.export(Model(),(torch.arange(4),torch.ones(4,32,device='cuda',dtype=torch.bfloat16),cache.table,torch.tensor([False])),
                               dynamic_shapes=({0:dim},{0:dim},None,None),strict=False)
        torch.export.save(ep,out/'export.pt2')
        (out/'graph.txt').write_text(str(ep.graph))
        path=torch._inductor.aoti_compile_and_package(ep,package_path=str(out/'model.pt2'))
        model=torch._inductor.aoti_load_package(path)
        results=[]
        for ids in ([1,2,3,4],[1,2,3,4],[1],[5],[1,2,3,6],[7,8,9,10,11,12,13,14]):
            x=torch.tensor(ids,device='cuda',dtype=torch.bfloat16)[:,None].expand(-1,32).contiguous()
            actual,ticket=model(torch.tensor(ids),x,cache.table,torch.tensor([False]))
            stats=torch.ops.history_single.retire(ticket).tolist()
            expected=x.float().sum(1)*256
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            results.append({'batch':len(ids),'stats':stats,'bit_exact':True})
            print('COMPILED_NATIVE_PASS',results[-1],flush=True)
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    print('COMPILED_NATIVE_COMPLETE',flush=True)


if __name__=='__main__':main()
