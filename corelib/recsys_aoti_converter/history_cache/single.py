# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""One end-to-end graph: common structural rewrite, native packing and cache ops."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.fx.experimental._config as symbolic_config
from torch.fx import Graph,GraphModule,Node
from torch.utils import _pytree as pytree

from .convert import CacheSpec,partition_history,_ancestors
from .slicing import TokenSlicer,dependencies,eliminate_common_subexpressions
from .precision import compile_options,precision_name,tensor_bytes,configure_math,check_math_policy
from . import single_ops
from .attention import replace_attention
from .review import tag_original,mark_nodes,write_review
from .dataflow_precision import validate_autocast,audit_dataflow
from .single_capture import export_dynamic


def rewrite(original,spec,*,sdpa=False,review_dir=None,source_modules=None,fold_attention=False):
    if fold_attention and not sdpa:
        raise ValueError('Attention query grouping requires SDPA replacement')
    part=partition_history(original,spec)
    from .layered import LayeredPlan, rewrite_layered
    if isinstance(part, LayeredPlan):
        return rewrite_layered(part,spec,sdpa=sdpa,review_dir=review_dir,source_modules=source_modules)
    module=original.module()
    if review_dir is not None:tag_original(module)
    nodes=list(module.graph.nodes)
    by_name={n.name:n for n in nodes}
    inputs=[n for n in nodes if n.op=='placeholder']
    history={inputs[i] for i in spec.history_input_indices}
    deps=dependencies(nodes)
    banks=part.report['banks']
    pair_nodes=[(by_name[b['key_node']],by_name[b['value_node']]) for b in banks]
    producer_ancestors=_ancestors([n for pair in pair_nodes for n in pair])
    positions={n:i for i,n in enumerate(nodes)}
    triggers={min(pair,key=positions.get):i for i,pair in enumerate(pair_nodes)}
    reads={}
    for i,pair in enumerate(pair_nodes):
        first=min((user for n in pair for user in n.users),key=positions.get)
        reads.setdefault(first,[]).append(i)
    graph,env=Graph(),{}
    ids=graph.placeholder('snapshot_ids')
    enabled=graph.placeholder('enable_offload')
    pools=[graph.placeholder(f'pool_{i}') for i in range(len(banks))]
    for n in inputs:
        env[n]=graph.placeholder(n.name)
        env[n].meta=dict(n.meta)
    ops=torch.ops.history_single
    prepare=graph.call_function(ops.prepare.default,(ids,env[inputs[0]],spec.length,len(banks)))
    import operator
    ticket,rows,slots=[graph.call_function(operator.getitem,(prepare,i)) for i in range(3)]
    packed=graph.call_function(ops.pack.default,([env[inputs[i]] for i in part.producer_input_indices],rows,slots,ticket))
    packed_env={inputs[index]:graph.call_function(operator.getitem,(packed,j)) for j,index in enumerate(part.producer_input_indices)}
    missing_slots=graph.call_function(operator.getitem,(packed,len(part.producer_input_indices)))
    if review_dir is not None:
        mark_nodes([n for n in graph.nodes if n not in env.values()],'request_prepare_pack')
    anchor=env[inputs[0]]
    insertions=[]
    fills={}
    for n in nodes:
        if n.op=='placeholder':continue
        if n in triggers:
            bank_index=triggers[n]
            key,value=pair_nodes[bank_index]
            prefix=banks[bank_index]['prefix_length']
            producer=TokenSlicer(deps,history,graph,dict(packed_env))
            previous=set(graph.nodes)
            new=[]
            for source in (key,value):
                selected=producer.select(source,2,0,prefix)
                if not selected.deps <= history:
                    raise ValueError('producer has request-specific inputs')
                item=selected.node
                if prefix<spec.length:
                    item=graph.call_function(torch.ops.aten.constant_pad_nd.default,(item,[0,0,0,spec.length-prefix],0.0))
                new.append(item)
            filled=graph.call_function(ops.fill.default,(pools[bank_index],missing_slots,*new,ticket,bank_index))
            fills[bank_index]=filled
            if review_dir is not None:
                mark_nodes([x for x in graph.nodes if x not in previous],f'bank_{bank_index:02d}_producer_fill',bank_index)
            insertions.append({'bank':bank_index,'original_trigger':n.name,
                               'prefix':prefix,'fill':filled.name})
        for bank_index in reads.get(n,[]):
            previous=set(graph.nodes)
            key,value=pair_nodes[bank_index]
            prefix=banks[bank_index]['prefix_length']
            filled=fills[bank_index]
            ready=graph.call_function(ops.onboard_wait.default,(ticket,filled,anchor))
            dense=graph.call_function(ops.gather.default,(pools[bank_index],slots,ready,bank_index))
            full=[graph.call_function(operator.getitem,(dense,i)) for i in range(2)]
            consumer=TokenSlicer(deps,history,graph,env)
            replacements=[]
            for source,whole in zip((key,value),full):
                if prefix<spec.length:
                    head=graph.call_function(torch.ops.aten.slice.Tensor,(whole,2,0,prefix))
                    tail=consumer.select(source,2,prefix,spec.length).node
                    replacements.append(graph.call_function(torch.ops.aten.cat.default,([head,tail],2)))
                else:replacements.append(whole)
            env[key],env[value]=replacements
            # Boundary metadata remains the full B/H/L/D contract. It also
            # lets later attention matching validate gathered K/V dtypes.
            for source,replacement in zip((key,value),replacements):
                replacement.meta=dict(source.meta)
            if review_dir is not None:
                mark_nodes([x for x in graph.nodes if x not in previous],f'bank_{bank_index:02d}_wait_gather_tail',bank_index)
            insertions[bank_index].update(original_first_read=n.name,wait=ready.name,
                                         gather=dense.name,wait_anchor=anchor.name)
        if n.op=='output':
            output=pytree.tree_leaves(n.args[0])
            if not all(isinstance(x,Node) and isinstance(x.meta.get('val'),torch.Tensor) for x in output):
                raise ValueError('Only tensor outputs supported')
            values=[env[x] for x in output]
            final=graph.call_function(ops.offload_launch.default,(ticket,values[-1],enabled))
            if review_dir is not None:mark_nodes([final],'request_offload')
            graph.output(tuple(values+[final]))
        elif n not in env and all(n not in pair for pair in pair_nodes):
            env[n]=graph.node_copy(n,lambda x:env[x])
        if n.op=='call_function' and n not in producer_ancestors and isinstance(n.meta.get('val'),torch.Tensor):
            anchor=env[n]
    eliminate_common_subexpressions(graph)
    converted=GraphModule(module,graph,'SingleHistoryModel')
    graph.eliminate_dead_code()
    graph.lint()
    converted.recompile()
    report=dict(part.report)
    report.update(execution='one end-to-end ExportedProgram/AOTI package; native pack/fill/wait/gather/offload',
                  insertion_sites=insertions,native_fill_gather_fused=False,
                  copied_model_sources=False,output_abi='original flattened outputs plus private submission ticket')
    report['attention']=replace_attention(converted) if sdpa else {'enabled':False,'matched':0}
    if fold_attention:
        from .attention_layout import fold_broadcast_attention
        report['attention_layout']=fold_broadcast_attention(converted)
    if sdpa and not report['attention']['matched']:
        raise ValueError('SDPA requested but no suitable attention computation matched')
    if review_dir is not None:
        report['review']=write_review(module,converted,report,review_dir,source_modules)
    return converted,report


def options(precision):
    result=compile_options(precision)
    if precision_name(precision)=='bf16':result['shape_padding']=False
    return result


def convert(original,spec,output,*,cache,input_tree,precision='bf16',tf32=False,compile=True,
            autocast=False,sdpa=False,review_diff=False,fold_attention=False):
    validate_autocast(precision,autocast)
    pure_bf16=precision_name(precision)=='bf16' and not autocast
    audit_dataflow(original,require_bf16=pure_bf16)
    math_policy=configure_math(precision,tf32)
    symbolic_config.backed_size_oblivious=True
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    single_ops.load(cache.project)
    single_ops.set_max_batch_size(cache.max_batch)
    source_path=output/'source_modules.json'
    source_modules=json.loads(source_path.read_text()) if source_path.exists() else None
    converted,report=rewrite(original,spec,sdpa=sdpa,
        review_dir=output/'review' if review_diff else None,source_modules=source_modules,fold_attention=fold_attention)
    if cache.banks!=len(report['banks']) or (cache.length,cache.heads,cache.head_dim)!=(spec.length,spec.heads,spec.head_dim):
        raise ValueError('Native cache geometry differs from conversion specification')
    if precision_name(cache.dtype)!=precision_name(precision):
        raise ValueError('Native cache precision differs from conversion precision')
    flat=list(original.example_inputs[0])
    if any(not x.is_contiguous() for x in flat):
        raise ValueError('The single-program input ABI requires contiguous tensors')
    batch_size=flat[0].shape[0]
    example=(torch.arange(100,100+batch_size,dtype=torch.int64),torch.tensor([False]),*cache.tables,*flat)
    def shapes(minimum):
        dim=torch.export.Dim('batch',min=minimum,max=cache.max_batch)
        return ({0:dim},None,*(None for _ in cache.tables),*({0:dim} for _ in flat))
    (output/'converted_before_export.py').write_text(converted.code)
    (output/'conversion_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with torch.no_grad():
        ep=export_dynamic(converted,example,shapes)
    dtype_report=audit_dataflow(ep,require_bf16=pure_bf16)
    (output/'converted_dtype_audit.json').write_text(json.dumps(dtype_report,indent=2)+'\n')
    torch.export.save(ep,output/'model_export.pt2')
    (output/'model_graph.txt').write_text(str(ep.graph))
    print('SINGLE_EXPORT_OK',ep.range_constraints,flush=True)
    compile_seconds=None
    if compile:
        start=time.perf_counter()
        torch._inductor.aoti_compile_and_package(ep,package_path=str(output/'model_aoti.pt2'),inductor_configs=options(precision))
        compile_seconds=time.perf_counter()-start
        print('SINGLE_COMPILE_OK',compile_seconds,flush=True)
    check_math_policy(tf32)
    digest=hashlib.sha256((str(original.graph)+str(spec)+precision_name(precision)+json.dumps(math_policy,sort_keys=True)+str(sdpa)+str(autocast)+('fold_attention' if fold_attention else '')).encode())
    for name,tensor in sorted(original.state_dict.items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor_bytes(tensor))
    manifest={'format_version':6,'precision':precision_name(precision),'autocast':autocast,'preserve_explicit_fp64':True,'tf32':tf32,'math_policy':math_policy,'batch_range':[1,cache.max_batch],
        'sdpa':sdpa,'attention_replacements':report['attention']['matched'],'review_diff':review_diff,
        'fold_attention':fold_attention,'attention_layout':report.get('attention_layout'),
        'spec':asdict(spec),'banks':report['banks'],'bank_count':len(report['banks']),
        'input_tree':pytree.treespec_dumps(input_tree),'output_tree':pytree.treespec_dumps(original.call_spec.out_spec),
        'input_shapes':[list(x.shape) for x in flat],'input_dtypes':[str(x.dtype) for x in flat],
        'namespace':digest.hexdigest(),'capacity':cache.capacity,
        'torch':torch.__version__,'cuda':torch.version.cuda,'device_capability':list(torch.cuda.get_device_capability()),
        'compile_seconds':compile_seconds,'compile_options':options(precision),'compiled':compile,
        'runtime':'serialized model execution per process; selectable native transfer owner; CUDA completion and protected transfer lifetime',
        'snapshot_contract':'caller-supplied immutable snapshot IDs; same ID must have identical history inputs',
        'ops':['history_single.prepare','history_single.pack','history_single.fill','history_single.onboard_wait','history_single.gather','history_single.offload_launch'],
        'limitations':['independent cache per instance','offload enabled explicitly per request','fixed history snapshots; no suffix append']}
    if 'history_value_input_indices' in report:
        manifest['history_value_input_indices']=report['history_value_input_indices']
        manifest['spec']['history_input_indices']=report['history_value_input_indices']
        manifest['layer_count']=report['layer_count']
    if 'context_guard_elision' in report:
        manifest['context_guard_elision']=report['context_guard_elision']
        if report['context_guard_elision']['elided']:
            # The snapshot contract covers history validity as well as values.
            # Optional fingerprint validation must retain both dependency sets.
            manifest['spec']['history_input_indices']=sorted(set(
                manifest['spec']['history_input_indices']) | set(
                report['context_guard_elision']['history_validity_input_indices']))
    if report.get('context_guard'):
        manifest['context_guard']=report['context_guard']
        manifest['snapshot_contract']+='; original total valid-token count must also remain fixed for that ID'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class SingleRunner:
    def __init__(self,bundle,cache,*,exported=False,verify_history=True,single_threaded=False):
        self.bundle=Path(bundle);self.cache=cache
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        self.input_tree=pytree.treespec_loads(self.manifest['input_tree'])
        self.output_tree=pytree.treespec_loads(self.manifest['output_tree'])
        m=self.manifest
        self.tf32=m.get('tf32',False)
        configure_math(m['precision'],self.tf32)
        if m['torch']!=torch.__version__ or tuple(m['device_capability'])!=torch.cuda.get_device_capability():
            raise ValueError('Recompile for the current PyTorch build and GPU architecture')
        geometry=(cache.length,cache.heads,cache.head_dim)
        expected_geometry=tuple(m['spec'][key] for key in ('length','heads','head_dim'))
        if (not cache.opened or cache.banks!=m['bank_count'] or cache.capacity!=m['capacity']
                or cache.max_batch!=m['batch_range'][1]
                or precision_name(cache.dtype)!=m['precision'] or geometry!=expected_geometry):
            raise ValueError('Native cache configuration differs from compiled bundle')
        namespace=getattr(cache,'single_namespace',m['namespace'])
        if namespace!=m['namespace']:raise ValueError('Different model bundles require separate cache runtimes')
        cache.single_namespace=m['namespace']
        self.verify_history=verify_history;self.fingerprints={}
        single_ops.load(cache.project)
        if self.manifest.get('context_guard') and not getattr(cache, 'single_context_guard_initialized', False):
            torch.ops.history_single.reset_context(torch.empty(0,dtype=torch.int64))
            cache.single_context_guard_initialized = True
        self.model=(torch.export.load(self.bundle/'model_export.pt2').module() if exported else
                    torch._inductor.aoti_load_package(str(self.bundle/'model_aoti.pt2'),run_single_threaded=single_threaded))
        self.failed=False;self.last_stats=None

    def validate_inputs(self,args,kwargs,snapshot_ids):
        flat,tree=pytree.tree_flatten((args,kwargs))
        if tree!=self.input_tree:raise ValueError('Original input tree changed')
        return self.validate_flat_inputs(flat,snapshot_ids)

    def validate_flat_inputs(self,flat,snapshot_ids,*,check_tensor_contract=True):
        """Validate flat inputs; trusted owners may reuse checked tensor storage.

        Disabling tensor metadata checks requires the same prevalidated tensor
        objects, shapes, dtypes and storage. Snapshot and runtime checks remain.
        """
        self.cache.assert_owner_thread()
        if self.failed:raise RuntimeError('Failed runner must be discarded')
        check_math_policy(self.tf32)
        if len(flat)!=len(self.manifest['input_shapes']):raise ValueError('Original input count changed')
        if check_tensor_contract and any(not isinstance(x,torch.Tensor) or x.ndim==0 for x in flat):raise ValueError('Batched tensor inputs required')
        batch=flat[0].shape[0]
        if not self.manifest['batch_range'][0]<=batch<=self.manifest['batch_range'][1] or len(snapshot_ids)!=batch:
            raise ValueError('Batch and snapshot IDs must match the compiled batch range')
        if check_tensor_contract:
            for x,shape,dtype in zip(flat,self.manifest['input_shapes'],self.manifest['input_dtypes']):
                if (x.device!=self.cache.table.device or not x.is_contiguous()
                        or list(x.shape[1:])!=shape[1:] or x.shape[0]!=batch or str(x.dtype)!=dtype):
                    raise ValueError('Input tensor contract mismatch')
        ids=torch.as_tensor(snapshot_ids,dtype=torch.int64,device='cpu')
        if ids.ndim!=1:raise ValueError('Snapshot IDs must be a one-dimensional integer sequence')
        return flat,ids

    def verify_snapshots(self,flat,ids):
        if self.verify_history:
            for row,identity in enumerate(ids.tolist()):
                digest=hashlib.sha256()
                for index in self.manifest['spec']['history_input_indices']:
                    digest.update(tensor_bytes(flat[index][row]))
                fingerprint=digest.hexdigest()
                previous=self.fingerprints.get(identity,fingerprint)
                if fingerprint!=previous:raise ValueError('immutable_history_changed: use a new snapshot identity')
                self.fingerprints[identity]=fingerprint

    def record_stats(self,ticket):
        stats=torch.ops.history_single.retire(ticket)
        self.last_stats=dict(zip(('unique_histories','misses','cpu_hits','pack_calls','wait_calls','gather_calls','bank_count','offloaded'),stats.tolist()))

    @torch.inference_mode()
    def __call__(self,*args,snapshot_ids,offload=False,**kwargs):
        flat,ids=self.validate_inputs(args,kwargs,snapshot_ids)
        return pytree.tree_unflatten(self._run_flat(flat,ids,offload),self.output_tree)

    @torch.inference_mode()
    def run_flat(self,flat,*,snapshot_ids,offload=False,check_tensor_contract=True):
        flat,ids=self.validate_flat_inputs(flat,snapshot_ids,check_tensor_contract=check_tensor_contract)
        return self._run_flat(flat,ids,offload)

    def _run_flat(self,flat,ids,offload):
        with self.cache.lock:
            self.verify_snapshots(flat,ids)
            try:
                result=self.model(ids,torch.tensor([offload]),*self.cache.tables,*flat)
                ticket=result[-1]
                self.record_stats(ticket)
            except BaseException:
                self.failed=self.cache.failed=True
                raise
        return list(result[:-1])
