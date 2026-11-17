# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, optional graph review artifacts; no model-name heuristics."""
import difflib
import inspect
import json
from pathlib import Path

import torch
from torch.fx import Node


def tag_original(module):
    for index, node in enumerate(module.graph.nodes):
        identity = f'o{index:05d}'
        node.meta.update(review_id=identity, review_origins=[identity],
                         review_original_name=node.name, review_role='original')


def merge_origins(target, source):
    for key in ('review_origins', 'review_banks'):
        if source.meta.get(key):
            target.meta[key] = sorted(set(target.meta.get(key, [])) | set(source.meta[key]))


def mark_nodes(nodes, role, bank=None):
    for index, node in enumerate(nodes):
        node.meta['review_id'] = f'{role}_{index:03d}'
        node.meta['review_role'] = role
        if bank is not None:
            node.meta['review_banks'] = [bank]


def source_catalog(model):
    result = {}
    for name, child in model.named_modules():
        if type(child).__module__.startswith('torch.'):
            continue
        try:
            _, line = inspect.getsourcelines(type(child).forward)
            result['model' + ('.' + name if name else '')] = {
                'class': type(child).__qualname__,
                'file': inspect.getsourcefile(type(child)), 'forward_line': line}
        except (TypeError, OSError):
            pass
    return result


def _ancestors(nodes):
    found, todo = set(), list(nodes)
    while todo:
        node = todo.pop()
        if node in found:
            continue
        found.add(node)
        todo.extend(node.all_input_nodes)
    return found


def _record(node):
    val = node.meta.get('val')
    result = {'id': node.meta.get('review_id', node.name), 'fx_name': node.name,
              'op': node.op, 'target': str(node.target),
              'origins': node.meta.get('review_origins', []),
              'banks': node.meta.get('review_banks', []),
              'role': node.meta.get('review_role', 'generated'),
              'inputs': [x.meta.get('review_id', x.name) for x in node.all_input_nodes]}
    if isinstance(val, torch.Tensor):
        result.update(shape=[str(x) for x in val.shape], dtype=str(val.dtype))
    for key in ('review_slice', 'stack_trace', 'nn_module_stack'):
        if key in node.meta:
            result[key] = str(node.meta[key])
    return result


def _render(nodes):
    def argument(x):
        if isinstance(x, Node):
            return x.meta.get('review_id', x.name)
        if isinstance(x, (tuple, list)):
            return '[' + ', '.join(argument(v) for v in x) + ']'
        if isinstance(x, dict):
            return '{' + ', '.join(f'{k}={argument(v)}' for k, v in x.items()) + '}'
        return repr(x)
    lines = []
    for node in nodes:
        r = _record(node)
        details = []
        if 'shape' in r:
            details.append(f"{r['dtype']}[{','.join(r['shape'])}]")
        if r.get('review_slice'):
            details.append(r['review_slice'])
        if r['role'] != 'original':
            details.append('role=' + r['role'])
            details.append('from=' + ','.join(r['origins']))
        text = f"{r['id']} = {r['target']}({argument(node.args)}"
        if node.kwargs:
            text += ', ' + argument(node.kwargs)
        text += ')'
        if details:
            text += '  # ' + '; '.join(details)
        lines.append(text + '\n')
    return lines


def write_review(before, after, report, output, catalog=None):
    """Emit stable-origin diffs, per-bank excerpts and complete node mappings."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old, new = list(before.graph.nodes), list(after.graph.nodes)
    old_names = {n.name: n for n in old}
    (output/'original_before_rewrite.py').write_text(before.code)
    (output/'converted_before_export.py').write_text(after.code)
    a, b = _render(old), _render(new)
    (output/'original.fx.txt').write_text(''.join(a))
    (output/'converted.fx.txt').write_text(''.join(b))
    (output/'full.diff').write_text(''.join(difflib.unified_diff(a,b,fromfile='original.fx',tofile='converted.fx',n=3)))
    records = [_record(n) for n in new]
    mappings = {n.meta['review_id']: {'original': _record(n), 'converted': []} for n in old}
    for r in records:
        for origin in r['origins']:
            if origin in mappings:
                mappings[origin]['converted'].append(r['id'])
    (output/'node_map.json').write_text(json.dumps({'original_to_converted': mappings,
                                                 'converted_nodes': records},indent=2)+'\n')
    lines = ['# Generated cache conversion review', '',
        'Generated deterministically by the converter. Original-node IDs are stable across the before/after views of this conversion; FX suffixes are not used for alignment. Sliced producers and fused attention carry original-node mappings. Deduplication merges origin lists; removed nodes have no surviving mapping.', '',
        'The Python files are full FX code. The annotated text and diffs group changes by cache bank. They do not reconstruct the original Python class/loop hierarchy. Shape comments on sliced nodes describe the original tensor; the slice annotation identifies the interval, and producers run on packed missing rows.', '',
        'Capture metadata can contain generated stack frames. Source locations below are obtained from the original model module catalog and parameter ownership when available; unavailable locations are not invented.', '',
        '[Full annotated diff](full.diff) · [Node mapping](node_map.json) · [Original code](original_before_rewrite.py) · [Converted code](converted_before_export.py)', '']
    bank_records = []
    for i, bank in enumerate(report['banks']):
        roots = [old_names[bank['key_node']], old_names[bank['value_node']]]
        selected = _ancestors(roots)
        # Include bounded downstream context, enough to expose layout and the
        # first attention uses without printing the entire unrolled decoder.
        frontier, seen = list(roots), set(roots)
        for _ in range(7):
            next_frontier = []
            for node in frontier:
                for user in node.users:
                    if user not in seen and len(seen) < 80:
                        seen.add(user); next_frontier.append(user)
            frontier = next_frontier
        selected |= seen
        origins = {n.meta['review_id'] for n in selected}
        after_selected = [n for n in new if
            i in n.meta.get('review_banks', []) or
            n.meta.get('review_role','').startswith('request_') or
            (not n.meta.get('review_banks') and bool(origins & set(n.meta.get('review_origins', []))))]
        aa, bb = _render([n for n in old if n in selected]), _render(after_selected)
        stem = f'bank_{i:02d}'
        (output/f'{stem}.before.txt').write_text(''.join(aa))
        (output/f'{stem}.after.txt').write_text(''.join(bb))
        (output/f'{stem}.diff').write_text(''.join(difflib.unified_diff(aa,bb,fromfile=f'{stem}.before',tofile=f'{stem}.after',n=3)))
        weights = sorted({str(n.target) for n in _ancestors(roots) if n.op == 'get_attr'})
        sources = {name: info for name, info in (catalog or {}).items()
                   if any(w.startswith(name+'.') for w in weights)}
        entry = {'bank': i, **bank, 'insertion': report['insertion_sites'][i],
                 'parameter_paths': weights, 'source_modules': sources}
        bank_records.append(entry)
        lines += [f'## Bank {i}', '',
            f"Original K/V: `{bank['key_node']}` / `{bank['value_node']}`. Reusable prefix: {bank['prefix_length']}; request tail: {bank['request_tail_length']}; storage: {bank['storage_length']}. History source offsets: {bank['history_source_offsets']}.", '',
            '```text',
            'prepare / pack missing raw inputs',
            f"  -> history-only producer ({bank['prefix_length']} rows) -> pad/fill bank {i}",
            '  -> onboard_wait -> gather for original batch',
            f"  -> cached prefix + request-local tail ({bank['request_tail_length']} rows)",
            '  -> original attention consumers (SDPA where matched)', '```', '',
            f'[Annotated diff]({stem}.diff) · [Before]({stem}.before.txt) · [After]({stem}.after.txt)', '']
        for name, info in sources.items():
            lines.append(f"- `{name}`: `{info['file']}:{info['forward_line']}` (`{info['class']}.forward`)")
        lines.append('')
    attention = report.get('attention', {})
    lines += ['## Attention replacement', '',
              f"SDPA enabled: {attention.get('enabled',False)}; matched: {attention.get('matched',0)}. Full replacement/skip details are in `../conversion_report.json`.", '']
    (output/'banks.json').write_text(json.dumps(bank_records,indent=2)+'\n')
    (output/'README.md').write_text('\n'.join(lines))
    return {'directory': str(output), 'banks': len(bank_records), 'original_nodes': len(old),
            'converted_nodes': len(new), 'mapping': 'node_map.json', 'index': 'README.md'}
