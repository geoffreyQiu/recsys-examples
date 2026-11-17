# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Check that review annotations are deterministic and leave graph code unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_cache.convert import CacheSpec
from history_cache.single import rewrite
from history_cache.single_ops import load


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir()) if p.is_file()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite evidence')
    load(Path(__file__).resolve().parents[1])
    ep = torch.export.load(a.bundle / 'original_export.pt2')
    spec = CacheSpec(**json.loads(a.spec.read_text()))
    source = a.bundle / 'source_modules.json'
    catalog = json.loads(source.read_text()) if source.exists() else None
    result = {'bundle': str(a.bundle), 'checks': []}
    with tempfile.TemporaryDirectory(prefix='review_artifacts_') as tmp:
        directory = Path(tmp)
        for sdpa in (False, True):
            plain, _ = rewrite(ep, spec, sdpa=sdpa)
            annotated, report = rewrite(ep, spec, sdpa=sdpa,
                                        review_dir=directory, source_modules=catalog)
            assert plain.code == annotated.code, 'Review flag changed executable graph code'
            first = hashes(directory)
            repeated, _ = rewrite(ep, spec, sdpa=sdpa,
                                  review_dir=directory, source_modules=catalog)
            assert repeated.code == annotated.code
            assert hashes(directory) == first, 'Review artifacts are nondeterministic'
            mapping = json.loads((directory / 'node_map.json').read_text())
            nodes = mapping['converted_nodes']
            identities = [n['id'] for n in nodes]
            assert len(identities) == len(set(identities)), 'Duplicate review IDs'
            for node in nodes:
                for origin in node['origins']:
                    assert node['id'] in mapping['original_to_converted'][origin]['converted']
            for bank in range(report['bank_count']):
                text = (directory / f'bank_{bank:02d}.after.txt').read_text()
                for op in ('fill', 'onboard_wait', 'gather'):
                    assert f'history_single.{op}.default' in text, (bank, op)
            result['checks'].append({'sdpa': sdpa, 'graph_unchanged': True,
                                     'deterministic': True, 'mapping_valid': True,
                                     'banks': report['bank_count'], 'files': len(first)})
            print('REVIEW_PASS', result['checks'][-1], flush=True)
    result['status'] = 'passed'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
