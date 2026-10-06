#!/usr/bin/env python3
"""Summarize fixed arrival qualification logs, retaining censored entries."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def spectral_bin(value):
    for bound, label in [(10, '<10'), (1e4, '10–1e4'), (1e8, '1e4–1e8')]:
        if value < bound:
            return label
    return '>=1e8'


def residual(method):
    geometry = method.get('geometry', [])
    if isinstance(geometry, dict):
        geometry = [geometry]
    values = [r['max_sin_squared'] for r in geometry
              if r.get('status') == 'measured' and 'max_sin_squared' in r]
    connector = method.get('connector') or {}
    if connector.get('status') == 'measured':
        values.append(connector['sin_squared'])
    return max(values) if values else None


def stats(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'median': statistics.median(values),
            'p90': values[min(len(values)-1, int(.9*len(values)))], 'max': values[-1]}


def failures(method):
    """Overlapping diagnostic counts; the recorded pass flag is authoritative."""
    if method.get('pass'):
        return []
    if method.get('status') != 'measured':
        return [method.get('reason', method.get('status', 'missing'))]
    out = []
    geometry = method.get('geometry', [])
    if isinstance(geometry, dict):
        geometry = [geometry]
    for r in geometry:
        if r.get('status') != 'measured':
            out.append('geometry:' + r.get('reason', r.get('status', 'missing')))
        if r.get('max_sin_squared', 0) > 1e-6:
            out.append('geometry:angular')
        for key in ('opposed_flow', 'orthogonal_flow', 'zero_tangents', 'stationary_samples', 'unvalidated_steps'):
            if r.get(key, 0):
                out.append('geometry:' + key)
        for key in ('binding_eps', 'join_eps'):
            if r.get(key, 0) > 64:
                out.append('geometry:' + key)
    r = method.get('connector') or {}
    if r.get('sin_squared', 0) > 1e-6:
        out.append('connector:angular')
    if r and (r.get('flow_alignment') != 1 or r.get('loss_direction') != 1):
        out.append('connector:orientation')
    if 'term' in method and method['term'] != 'capture':
        out.append('term:' + method['term'])
    return sorted(set(out)) or ['other_check']


def summarize_entries(entries, methods):
    result = {'entries': len(entries), 'fate_forced': sum(e['forced'] for e in entries), 'methods': {}}
    for method in methods:
        eligible = [e for e in entries if method != 'current_connector' or e['cohort'] != 'corpus']
        rows = [e['methods'].get(method, {}) for e in eligible]
        result['methods'][method] = {
            'eligible': len(rows), 'pass': sum(bool(r.get('pass')) for r in rows),
            'pass_and_forced': sum(bool(e['methods'].get(method, {}).get('pass')) and e['forced'] for e in eligible),
            'statuses': dict(Counter(r.get('status', 'missing') for r in rows)),
            'reasons': dict(Counter(r.get('reason', '') for r in rows if r.get('status') in ('error', 'timeout'))),
            'failure_causes_overlapping': dict(Counter(cause for r in rows for cause in failures(r))),
            'build_seconds': stats(r.get('build_seconds') for r in rows),
            'total_seconds': stats(r.get('seconds') for r in rows),
            'check_seconds': stats(r['seconds']-r['build_seconds'] for r in rows if 'build_seconds' in r),
            'measured_residual': stats(residual(r) for r in rows),
            'modes': dict(Counter(r['mode'] for r in rows if 'mode' in r)),
        }
    result['paired'] = {}
    for baseline in ('current_connector', 'raw_arrival'):
        counts = Counter()
        for e in entries:
            if baseline == 'current_connector' and e['cohort'] == 'corpus':
                continue
            old = bool(e['methods'].get(baseline, {}).get('pass'))
            new = bool(e['methods'].get('analytic', {}).get('pass'))
            counts['both' if old and new else 'baseline_only' if old else 'analytic_only' if new else 'neither'] += 1
        result['paired'][baseline] = {k: counts[k] for k in ('both', 'analytic_only', 'baseline_only', 'neither')}
    return result


def summarize(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    completed = {r['index']: r for r in read_rows(directory/'completion.jsonl')}
    entries = []
    models = []
    for index, spec in enumerate(manifest['specs']):
        rows = read_rows(directory/f'{index:04d}.jsonl')
        model = next((r for r in rows if r['event'] == 'model'), {})
        state = model.get('status', 'missing')
        models.append({'job': index, **spec, 'model_status': state,
                       'completion': completed.get(index, {}), 'model': model,
                       'ended': any(r['event'] == 'end' for r in rows),
                       'refusals': [r for r in rows if r['event'] == 'entry_refused']})
        by_entry = {}
        for row in rows:
            if row['event'] in ('entry', 'entry_refused'):
                e = by_entry.setdefault(row['entry'], {'job': index, **spec, 'entry': row['entry'], 'forced': False, 'methods': {}})
                if row['event'] == 'entry':
                    e.update({k: v for k, v in row.items() if k != 'event'})
                else:
                    e['refusal'] = row['reason']
            elif row['event'] == 'method':
                by_entry[row['entry']]['methods'][row['method']] = row
            elif row['event'] == 'fate':
                by_entry[row['entry']]['forced'] = row.get('forced', False)
                by_entry[row['entry']]['fate_status'] = row.get('status')
        # Retain entries announced by model construction but lost to process failure.
        for missing in range(model.get('entries', 0)-len(by_entry)):
            by_entry[f'missing-{missing}'] = {'job': index, **spec, 'entry': f'missing-{missing}',
                                             'forced': False, 'methods': {}, 'refusal': 'unprocessed'}
        entries.extend(by_entry.values())
    methods = ('current_connector', 'raw_arrival', 'analytic')
    result = {'jobs': len(models), 'completed': len(completed),
              'model_statuses': dict(Counter(m['model_status'] for m in models)),
              'models_without_arrivals': sum(m['model_status'] == 'constructed' and not m['model'].get('entries') for m in models),
              'worker_timeouts': sum(r.get('status') == 'worker_timeout' for r in completed.values()),
              'worker_nonzero': sum(r.get('returncode', 0) != 0 for r in completed.values()),
              'entry_refusals': dict(Counter(e['refusal'] for e in entries if 'refusal' in e)),
              'all': summarize_entries(entries, methods), 'cohorts': {}, 'spectral_bins': {}}
    for cohort in ('random', 'directed', 'zoo', 'corpus'):
        subset = [e for e in entries if e['cohort'] == cohort]
        result['cohorts'][cohort] = summarize_entries(subset, methods)
        result['cohorts'][cohort]['model_statuses'] = dict(Counter(m['model_status'] for m in models if m['cohort'] == cohort))
    for label in ('<10', '10–1e4', '1e4–1e8', '>=1e8', 'unknown'):
        subset = [e for e in entries if (spectral_bin(e['spectral_ratio']) if 'spectral_ratio' in e else 'unknown') == label]
        result['spectral_bins'][label] = summarize_entries(subset, methods)
    result['regressions'] = [{k: v for k, v in e.items() if k != 'methods'} | {'methods': e['methods']}
                             for e in entries if not e['methods'].get('analytic', {}).get('pass')
                             and any(e['methods'].get(m, {}).get('pass') for m in methods[:2])]
    repo = Path(__file__).resolve().parents[1]
    result['source_hash_mismatches'] = [name for name, digest in manifest['sha256'].items()
                                       if not (repo/name).exists() or hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest]
    return result, entries, models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result, entries, models = summarize(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'entries.json').write_text(json.dumps(entries, indent=2))
    (args.output/'models.json').write_text(json.dumps(models, indent=2))
    fields = ['job', 'cohort', 'seed', 'name', 'entry', 'spectral_ratio', 'distance', 'forced', 'method', 'status', 'pass', 'residual', 'build_seconds', 'seconds', 'reason']
    with (args.output/'entries.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            for method in ('current_connector', 'raw_arrival', 'analytic'):
                if method == 'current_connector' and e['cohort'] == 'corpus':
                    continue
                r = e['methods'].get(method, {})
                row = {k: e.get(k, '') for k in fields[:8]}
                row.update({k: r.get(k, '') for k in ('status', 'pass', 'build_seconds', 'seconds', 'reason')})
                row.update(method=method, residual=residual(r), status=r.get('status', 'missing'), **{'pass': bool(r.get('pass'))})
                writer.writerow(row)
    print(json.dumps({k: v for k, v in result.items() if k not in ('regressions', 'spectral_bins')}, indent=2))


if __name__ == '__main__':
    main()
