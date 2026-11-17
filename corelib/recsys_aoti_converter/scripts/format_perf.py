# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Format generic benchmark_aoti.py JSON as a Markdown matrix."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--title', default='Sequential direct inference')
    p.add_argument('--digits', type=int, default=4, choices=range(1, 7))
    a = p.parse_args()
    result = json.loads(a.input.read_text())
    if not result.get('completed') or not result.get('perf'):
        p.error('The input must contain a completed benchmark')
    batches = result['batches']
    fused = 'attention_fusion_aoti' in result['perf'][str(batches[0])]
    columns = [('baseline_aoti', 'Baseline AOTI¹')]
    if fused:
        columns += [('attention_fusion_aoti', 'Attention fusion AOTI²'),
                    ('gpu_kv_reuse_aoti', 'Fusion + GPU KV reuse AOTI³'),
                    ('cpu_kv_reuse_aoti', 'Fusion + CPU KV reuse AOTI⁴')]
    else:
        columns += [('gpu_kv_reuse_aoti', 'GPU KV reuse AOTI²'),
                    ('cpu_kv_reuse_aoti', 'CPU KV reuse AOTI³')]
    capability = result.get('capability')
    arch = f' (SM{capability[0]}{capability[1]})' if capability else ''
    lines = [f"## {a.title}", '',
             f"{result['gpu']}{arch}; {result['precision'].upper()}, "
             f"autocast {'on' if result.get('autocast') else 'off'}. "
             'Median completed-forward latency in milliseconds; batches run sequentially.', '',
             '| Batch size | ' + ' | '.join(label for _, label in columns) + ' |',
             '|' + '---:|' * (len(columns) + 1)]
    for batch in batches:
        row = result['perf'][str(batch)]
        values = [f"{row[key]['median_ms']:.{a.digits}f}" for key, _ in columns]
        lines.append(f'| {batch} | ' + ' | '.join(values) + ' |')
    lines += ['', '- ¹ Original model without KV caching, exported with `torch.export` '
              'and compiled with PyTorch AOTInductor.']
    if fused:
        lines += ['- ² Converter attention fusion via SDPA, without KV reuse.']
    gpu_mark, cpu_mark = ('³', '⁴') if fused else ('²', '³')
    lines += [f'- {gpu_mark} All requested histories are GPU KV cache hits.',
              f'- {cpu_mark} All requested histories are CPU KV cache hits; onboarding and waits are included.',
              '', 'Every column uses an AOTI compiled graph. Startup, compilation, CPU '
              'publication and GPU eviction are excluded. No HTTP/Triton round trips.']
    first = result['perf'][str(batches[0])]['baseline_aoti']
    lines += [f"{first['warmup']} warmups and {first['repeats']} samples per path/batch; "
              'CUDA-synchronized completed-request wall time.', '']
    markdown = '\n'.join(lines)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(markdown)
    print(markdown, end='')


if __name__ == '__main__':
    main()
