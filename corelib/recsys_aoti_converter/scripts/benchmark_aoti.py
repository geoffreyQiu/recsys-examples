# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Offline AOTI latency matrix from captured inputs; no numerical comparisons."""
import argparse
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('AOTI_RUNTIME_CHECK_INPUTS', '0')
import torch
from torch.utils import _pytree as pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.native import NativeCache
from history_cache.precision import DTYPES, configure_math
from history_cache.single import SingleRunner


def timed(fn, prepare, warmup, repeats):
    samples = []
    for i in range(warmup + repeats):
        prepare()
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        if i >= warmup:
            samples.append(elapsed)
        del output
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'warmup': warmup, 'repeats': repeats}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--output', type=Path)
    p.add_argument('--owner-loop', action=argparse.BooleanOptionalAction, default=True,
                   help='Use the shared corelib cache owner (default)')
    a = p.parse_args()
    if a.warmup < 1 or a.repeats < 1 or any(b < 1 for b in a.batches):
        p.error('Use positive warmup/repeats and batch sizes')
    if len(set(a.batches)) != len(a.batches):
        p.error('Batch sizes must be unique')
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    m = json.loads((a.bundle / 'manifest.json').read_text())
    max_batch = m.get('batch_range', [1, 8])[1]
    if any(batch > max_batch for batch in a.batches):
        p.error(f'Batch sizes must be within the exported limit: {max_batch}')
    configure_math(m['precision'], m.get('tf32', False))
    exported = torch.export.load(a.bundle / 'original_export.pt2')
    base = list(exported.example_inputs[0])
    del exported
    controls = {'baseline_aoti': torch._inductor.aoti_load_package(
        str(a.bundle / 'uncached_aoti.pt2'))}
    if m.get('sdpa', False):
        controls['attention_fusion_aoti'] = torch._inductor.aoti_load_package(
            str(a.bundle / 'uncached_sdpa_aoti.pt2'))
    tree = pt.treespec_loads(m['input_tree'])
    result = {
        'completed': False, 'precision': m['precision'],
        'autocast': m.get('autocast', False), 'tf32': m.get('tf32', False),
        'sdpa': m.get('sdpa', False), 'cpu_threads': 4, 'owner_loop': a.owner_loop,
        'gpu': torch.cuda.get_device_name(),
        'capability': list(torch.cuda.get_device_capability()),
        'nv_gpu': os.environ.get('NV_GPU'), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'sequential direct inference; no concurrency or numerical comparisons',
        'timing_contract': 'CUDA-synchronized completed wall latency; input preparation, compilation, CPU publication and GPU eviction excluded; wrapper/cache controls and onboarding/waits included; history hashing excluded',
        'bundle': str(a.bundle), 'batches': a.batches, 'perf': {}, 'routes': {},
    }
    output = a.output or a.bundle / 'performance.json'
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')

    save()
    spec = m['spec']
    with NativeCache(root, a.reference, capacity=m['capacity'], length=spec['length'],
                     heads=spec['heads'], head_dim=spec['head_dim'],
                     banks=m['bank_count'], dtype=DTYPES[m['precision']],
                     max_batch=m.get('batch_range', [1, 8])[1], owner_loop=a.owner_loop) as cache:
        result['cache_runtime'] = dict(owner_loop=cache.owner_loop,
            batch_lookup=cache.batch_lookup, fast_metadata=cache.fast_metadata,
            resident_fastpath=cache.resident_fastpath,
            onboard_child_tasks=bool(cache.config['onboard_child_tasks']), cuda_graph=False)
        save()
        runner = SingleRunner(a.bundle, cache, verify_history=False)
        for batch in a.batches:
            flat = [x.index_select(0, torch.arange(batch, device=x.device) % x.shape[0])
                    for x in base]
            args, kwargs = pt.tree_unflatten(flat, tree)
            ids = list(range(100000 + 100 * batch, 100000 + 101 * batch))

            def invoke(offload=False):
                return runner(*args, **kwargs, snapshot_ids=ids, offload=offload)

            # Publish immutable snapshots before timing either all-hit path.
            invoke(True)
            cache.flush()
            times = {}
            for name, model in controls.items():
                times[name] = timed(
                    lambda: model(*pt.tree_flatten((args, kwargs))[0]),
                    lambda: None, a.warmup, a.repeats)
            times['gpu_kv_reuse_aoti'] = timed(invoke, lambda: None, a.warmup, a.repeats)
            gpu_route = dict(runner.last_stats)
            times['cpu_kv_reuse_aoti'] = timed(
                invoke, lambda: cache.evict(ids), a.warmup, a.repeats)
            cpu_route = dict(runner.last_stats)
            if (gpu_route['misses'] != 0 or gpu_route['cpu_hits'] != 0
                    or cpu_route['misses'] != 0 or cpu_route['cpu_hits'] != batch):
                raise RuntimeError(f'Expected all-hit paths: GPU={gpu_route}, CPU={cpu_route}')
            result['routes'][str(batch)] = {'gpu': gpu_route, 'cpu': cpu_route}
            result['perf'][str(batch)] = times
            save()
            print('PERF', batch, {k: round(v['median_ms'], 4) for k, v in times.items()}, flush=True)
    result['completed'] = True
    save()
    print('COMPLETED', str(output), flush=True)


if __name__ == '__main__':
    main()
