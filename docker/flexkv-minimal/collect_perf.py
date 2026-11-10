#!/usr/bin/env python3
"""Collect saved logs without GPU execution; keep an append-only performance history."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s,]+)', line))


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def collect(text, run_id, mode, exit_code, log_path, image, gpus, profile):
    configs = [fields(line) for line in text.splitlines() if line.startswith('[Config] history_len=')]
    cfg = configs[-1] if configs else {}
    samples = [fields(line) for line in text.splitlines() if line.startswith('[latency_sample]')]
    # An imported terminal transcript may include multiple scenarios.
    samples = [s for s in samples if s.get('scenario') == ('ssd_hit' if mode == 'ssd_layerwise' else mode)]
    vals = [float(s['ms']) for s in samples]
    expected = {'gpu_hit': 'Scenario1', 'cpu_hit': 'Scenario2', 'ssd_hit': 'Scenario3', 'ssd_layerwise': 'Scenario3'}.get(mode)
    completed = (bool(samples) and '[latency] scenario=no_cache' in text) if mode == 'no_cache' else bool(expected and f'[{expected}] timed run completed' in text)
    if mode == 'preflight':
        completed = 'PASS: GPU operation and benchmark imports' in text
    warnings = []
    if "object has no attribute 'suppress'" in text:
        warnings.append('FlexKV destructor error after shutdown')
    if 'Producer process has been terminated before all shared CUDA tensors released' in text:
        warnings.append('CUDA IPC cleanup warning')
    for line in text.splitlines():
        if re.match(r'^(?:[\w.]*Error|[\w.]*Exception|KeyboardInterrupt):', line):
            warnings.append(line[:300])
        if re.search(r'fail.open|fallback|status=(?:fail|error)|Traceback', line, re.I):
            if 'Traceback' not in line:  # destructor trace recorded above
                warnings.append(line[:300])
    record = {
        'run_id': run_id, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': mode, 'status': 'completed' if completed and exit_code in (None, 0) else 'failed_or_incomplete',
        'exit_code': exit_code, 'config': cfg, 'image': image, 'gpus': gpus,
        'profile': profile, 'log': str(log_path), 'samples_ms': vals,
        'sync_cache_alloc': cfg.get('sync_cache_alloc') == 'True',
        'allocator_impl': cfg.get('allocator_impl', 'unmarked'),
        'metric': 'synchronized_forward_wall_ms' if vals else None,
        'sample_count': len(vals), 'warnings': sorted(set(warnings)),
        'log_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }
    if vals:
        record.update(mean_ms=statistics.mean(vals), median_ms=statistics.median(vals),
                      p95_ms=percentile(vals, 95), p99_ms=percentile(vals, 99),
                      min_ms=min(vals), max_ms=max(vals), stddev_ms=statistics.pstdev(vals))
    record['assessment'] = 'preliminary (<30 samples)' if 0 < len(vals) < 30 else ('measured; inspect warnings' if vals else 'completion only; latency unavailable')
    if record['status'] == 'failed_or_incomplete':
        record['assessment'] = 'failed/incomplete; any partial samples are not valid benchmark results'
    # Transfer samples are components, not inference latency; retain direction.
    transfers = []
    for line in text.splitlines():
        if '[FlexKV-IO]' in line and 'operation=transfer' in line and 'status=success' in line:
            f = fields(line)
            transfers.append({k: f[k] for k in ('direction', 'blocks', 'data_size', 'transfer_time', 'bandwidth', 'op_id', 'graph_id') if k in f})
    if cfg.get('sync_cache_alloc') == 'True':
        record['assessment'] = 'diagnostic allocation synchronization; ' + record['assessment']
        record['warnings'].append('Diagnostic CPU wait before cache allocation changes scheduling and latency')
    record['transfer_samples'] = transfers
    metadata_errors = []
    for line in text.splitlines():
        if line.startswith('[CacheMetadataError] '):
            try:
                metadata_errors.append(json.loads(line.split(' ', 1)[1]))
            except json.JSONDecodeError:
                metadata_errors.append({'unparsed': line})
    record['cache_metadata_errors'] = metadata_errors
    return record, samples


def render(records):
    lines = ['# Running benchmark results — RTX PRO 5500', '',
             'Updated automatically from saved logs. UTC timestamps identify collection, not necessarily execution.', '',
             'Forward latency includes the model call and its completion on CUDA; request construction and cache eviction are outside the timer. Each iteration is synchronized, so this is sequential latency, not pipelined throughput. Transfer timings are separate components. Profiling can add overhead. Fewer than 30 samples are preliminary; more samples alone do not establish steady state or correctness.', '',
             '| Run | Case | Graphs | Profiled | N | Mean ms | P50 ms | P95 ms | Min–max ms | Status |',
             '|---|---|---|---|---:|---:|---:|---:|---|---|']
    for r in records:
        cfg = r.get('config', {})
        graph = {'True':'off', 'False':'on'}.get(cfg.get('disable_cudagraph'), '?')
        val = lambda key: f"{r[key]:.4f}" if key in r else '—'
        span = f"{r['min_ms']:.4f}–{r['max_ms']:.4f}" if 'min_ms' in r else '—'
        status = r['status'] + ('; warnings' if r.get('warnings') else '')
        if cfg.get('sync_cache_alloc') == 'True':
            status += '; alloc-sync diagnostic'
        lines.append(f"| {r['run_id']} | {r['mode']} | {graph} | {'yes' if r.get('profile') else 'no'} | {r['sample_count']} | {val('mean_ms')} | {val('median_ms')} | {val('p95_ms')} | {span} | {status} |")
    lines += ['', '## Run details', '']
    for r in records:
        lines += [f"### {r['run_id']}", '', f"- Assessment: {r['assessment']}.",
                  f"- Source log: `{r['log']}`; exit code: `{r['exit_code']}` (null means imported transcript, exit status not captured).",
                  f"- Config: `{json.dumps(r.get('config', {}), sort_keys=True)}`.",
                  f"- Image: `{r.get('image', '')}`; GPU selection: `{r.get('gpus', '')}`; profiled: `{r.get('profile', False)}`.",
                  f"- Transfer log samples: {len(r.get('transfer_samples', []))}; see JSONL/run JSON for individual values. These may include warmup transfers."]
        lines += [f'- Warning: {w}' for w in r.get('warnings', [])]
        lines += [f'- Cache metadata error: `{json.dumps(e, sort_keys=True)}`.' for e in r.get('cache_metadata_errors', [])]
        lines += ['']
    return '\n'.join(lines).rstrip() + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--mode', required=True, choices=['preflight','no_cache','gpu_hit','cpu_hit','ssd_hit','ssd_layerwise'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--exit-code', type=int)
    parser.add_argument('--image', default='')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    text = args.log.read_text(errors='replace')
    # Split transcripts at the per-scenario config; retain only the requested case.
    chunks = re.split(r'(?=^\[Config\] history_len=)', text, flags=re.M)
    desired = 'ssd_hit' if args.mode == 'ssd_layerwise' else args.mode
    selected = [c for c in chunks if re.search(rf'^\[Config\].*scenarios={desired}(?:\s|$)', c, re.M)]
    if selected:
        text = selected[-1]
    record, samples = collect(text, args.run_id, args.mode, args.exit_code, args.log.resolve(), args.image, args.gpus, args.profile)
    if args.run_dir:
        for filename, key in [('image-id.txt','image_id'), ('source-revision.txt','source_revision'), ('benchmark.sha256','benchmark_sha256'), ('command.txt','command'), ('gpus.csv','gpu_metadata_csv')]:
            path = args.run_dir / filename
            if path.is_file():
                record[key] = path.read_text(errors='replace').strip()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    with (args.record_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history = args.record_dir / 'runs.jsonl'
        records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()] if history.exists() else []
        seed = Path(__file__).with_name('perf-seed.jsonl')
        seeded = [json.loads(line) for line in seed.read_text().splitlines() if line.strip()] if seed.exists() else []
        known_ids = {r['run_id'] for r in records}
        missing = [r for r in seeded if r['run_id'] not in known_ids]
        if missing:
            with history.open('a') as out:
                for entry in missing:
                    out.write(json.dumps(entry, sort_keys=True) + '\n')
            records.extend(missing)
        old = next((r for r in records if r['run_id'] == record['run_id']), None)
        if old and old['log_sha256'] != record['log_sha256']:
            raise SystemExit('Run ID already exists with different log content; choose a new ID.')
        if not old:
            with history.open('a') as out:
                out.write(json.dumps(record, sort_keys=True) + '\n')
            records.append(record)
        report = args.record_dir / 'bench-results.md'
        temp = report.with_suffix('.md.tmp')
        temp.write_text(render(records))
        temp.replace(report)
        columns = ['run_id','mode','status','sample_count','mean_ms','median_ms','p95_ms','p99_ms','min_ms','max_ms','stddev_ms','metric','exit_code','profile','sync_cache_alloc','allocator_impl','log']
        with (args.record_dir / 'summary.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
    if args.run_dir:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        with (args.run_dir / 'samples.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=['scenario','batch_size','iter','ms','cuda_ms'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(samples)
    print(f"[perf] {record['mode']}: {record['status']}; N={record['sample_count']}" + (f" mean={record['mean_ms']:.4f} ms p95={record['p95_ms']:.4f} ms" if record['sample_count'] else '; no inference latency samples'))
    print(f"[perf] Running record: {args.record_dir / 'bench-results.md'}")


if __name__ == '__main__':
    main()
