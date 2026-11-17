# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic repeated-stack fixture for fixed history KV reuse."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('AOTI_RUNTIME_CHECK_INPUTS', '0')
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.convert import CacheSpec, partition_history
from history_cache.single_capture import capture
from history_cache.single import convert, SingleRunner
from history_cache.native import NativeCache
from history_cache.precision import configure_math, TOLERANCES, compare
from history_cache.slicing import ConversionError


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(128)
        self.uvqk = torch.nn.Linear(128, 512)
        self.norm2 = torch.nn.LayerNorm(64)
        self.linear = torch.nn.Linear(128, 128)

    def forward(self, x, valid, mask):
        u, v, q, k = self.uvqk(self.norm1(x)).split([128]*4, -1)
        u = F.silu(u.reshape(-1, 21, 2, 64).contiguous())
        v = v.reshape(-1, 21, 2, 64).contiguous()
        q = q.reshape(-1, 21, 2, 64).contiguous()
        k = k.reshape(-1, 21, 2, 64).contiguous()
        w = F.silu(torch.einsum('bqhd,bkhd->bhqk', q, k) * .125)
        w = w / valid.sum(1).clamp(min=1).reshape(-1, 1, 1, 1)
        w = w * mask.unsqueeze(1)
        a = torch.einsum('bhqk,bkhv->bhqv', w, v)
        g = u.transpose(1, 2) * self.norm2(a)
        return self.linear(g.transpose(1, 2).reshape(-1, 21, 128)) + x


class RepeatedStack(torch.nn.Module):
    def __init__(self, unsafe=False):
        super().__init__()
        self.unsafe = unsafe
        self.embed_history = torch.nn.Linear(7, 128)
        self.embed_candidate = torch.nn.Linear(9, 128)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])
        mask = torch.zeros(21, 21)
        mask[:5, :5] = torch.eye(5)
        mask[:5, 5:] = 1
        mask[5:, 5:] = torch.ones(16, 16).triu()
        if unsafe == 'mask':
            mask[6, 0] = 1
        self.register_buffer('structural_mask', mask)

    def forward(self, history, candidates, validity):
        if self.unsafe == 'batch':
            history = history + history.mean(0, keepdim=True)
        if self.unsafe == 'request':
            history = history + candidates.mean((1,2,3)).reshape(-1,1,1)
        valid = validity.to(history.dtype)
        mask = self.structural_mask.unsqueeze(0).repeat(history.shape[0], 1, 1)
        mask = mask * (valid.unsqueeze(-1) @ valid.unsqueeze(1))
        outputs = []
        for step in range(3):
            # Preprocessing can combine fields before selecting history. The
            # converter must remove the discarded candidate dependencies.
            combined = torch.cat([candidates[:, 0, :, :7], history], dim=1)
            h = self.embed_history(combined[:, 5:])
            c = self.embed_candidate(candidates[:, step])
            x = torch.cat([c, h], 1)
            for layer in self.layers:
                x = layer(x, valid, mask)
            outputs.append(x[:, :5])
        return torch.stack(outputs, 1)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(341)
    torch.set_num_threads(4)
    configure_math(a.precision, False)
    model = RepeatedStack().cuda().eval()
    inputs = (torch.randn(4,16,7,device='cuda'), torch.randn(4,3,5,9,device='cuda'),
              torch.ones(4,21,dtype=torch.bool,device='cuda'))
    ep, tree = capture(model, inputs, a.precision)
    (a.output/'original_graph.txt').write_text(str(ep.graph))
    print('EINSUMS',sum(str(n.target)=='aten.einsum.default' for n in ep.graph.nodes),flush=True)
    spec = CacheSpec((0,),16,2,64,history_start=5)
    part = partition_history(ep,spec)
    (a.output/'analysis.json').write_text(json.dumps(part.report,indent=2))
    print('LAYERS',part.banks,'PASSES',part.report['stack_passes'],flush=True)
    diagnostics = {}
    for unsafe in ('mask', 'batch', 'request'):
        bad, _ = capture(RepeatedStack(unsafe).cuda().eval(), inputs, a.precision)
        try:
            partition_history(bad, spec)
        except ConversionError as error:
            diagnostics[unsafe] = str(error)
        else:
            raise AssertionError('Unsafe history reuse accepted: ' + unsafe)
    if a.analyze_only:
        return
    root=Path(__file__).resolve().parents[1]
    dtype=torch.float32 if a.precision=='fp32' else torch.bfloat16
    cases = []
    with NativeCache(root,a.reference,length=16,heads=2,head_dim=64,banks=3,dtype=dtype) as cache:
        convert(ep,spec,a.output,cache=cache,input_tree=tree,precision=a.precision)
        runner=SingleRunner(a.output,cache)
        plain=torch.fx.GraphModule(ep.module(),ep.module().graph)
        base=ep.example_inputs[0]
        for batch in (1,2,4,8):
            current=tuple(x.index_select(0,torch.arange(batch,device='cuda')%4) for x in base)
            ids=list(range(100*batch,100*batch+batch))
            for label in ('miss','hit','mix','cpu'):
                if label=='mix':
                    ids=[i+10000 if j%2 else i for j,i in enumerate(ids)]
                if label=='cpu':
                    runner(*current,snapshot_ids=ids,offload=True)
                    cache.flush();cache.evict(ids)
                actual=runner(*current,snapshot_ids=ids)
                with torch.no_grad():expected=plain(*current)
                comparison=compare(actual,expected,tolerance=TOLERANCES[a.precision])
                if a.precision=='fp32':
                    torch.testing.assert_close(actual,expected,**TOLERANCES[a.precision])
                assert runner.last_stats['gather_calls']==3
                cases.append({'batch':batch,'case':label,'comparison':comparison,'stats':runner.last_stats.copy()})
                print('CASE',batch,label,comparison['numeric_within_tolerance'],comparison['max_abs'],runner.last_stats,flush=True)
            before=[x.clone() for x in cache.tables]
            changed=(current[0],current[1]+.125,current[2])
            got=runner(*changed,snapshot_ids=ids)
            assert runner.last_stats['misses']==0
            # Unallocated slots may contain arbitrary BF16/NaN bit patterns.
            assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(before,cache.tables))
            runner(*changed,snapshot_ids=ids,offload=True);cache.flush();cache.evict(ids)
            restored=runner(*changed,snapshot_ids=ids)
            assert torch.equal(got,restored)
        # Same total count, different H validity must still invalidate a snapshot.
        check=list(x[:1].clone() for x in base)
        check[2][0,5]=False
        runner(*check,snapshot_ids=[999999])
        check[2][0,5]=True;check[2][0,0]=False
        try:
            runner(*check,snapshot_ids=[999999])
        except RuntimeError as error:
            assert 'snapshot_context_changed' in str(error)
            diagnostics['history_validity_guard']=True
        else:
            raise AssertionError('Changed history validity accepted')
    result={'completed':True,'precision':a.precision,'cases':cases,'negative_tests':diagnostics,
            'status':'passed' if all(c['comparison']['numeric_within_tolerance'] for c in cases) else 'numerical_differences',
            'immutable_pools_and_exact_cpu_restore':True}
    (a.output/'test_summary.json').write_text(json.dumps(result,indent=2))
    print('LAYERED_FIXTURE_COMPLETED',a.precision,result['status'],flush=True)


if __name__=='__main__':main()
