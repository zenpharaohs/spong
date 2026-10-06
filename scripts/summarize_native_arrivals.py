#!/usr/bin/env python3
"""Join a native replay with explicitly selected corrective reruns."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def rows(directory):
    result = {}
    for p in sorted(directory.glob('[0-9]*.jsonl')):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r['event'] == 'arrival':
                key = (r['job'], r['entry'])
                if key in result:
                    raise ValueError(f'duplicate entry {key}')
                result[key] = r
    return result


def median(values):
    a = list(values)
    return statistics.median(a) if a else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('replay', type=Path)
    ap.add_argument('--correction', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    entries = {(e['job'], e['entry']): e for e in json.loads(args.source.read_text())}
    measured = rows(args.replay)
    if set(measured) != set(entries):
        raise ValueError('incomplete primary replay')
    corrected = {}
    if args.correction:
        corrected = rows(args.correction)
        expected = {k for k, r in measured.items() if '4300 digits' in r.get('reason', '')}
        if set(corrected) != expected:
            raise ValueError('corrective replay must cover every serialization failure exactly')
        measured.update(corrected)
    joined = []
    for key, r in measured.items():
        e = entries[key]
        g = r.get('geometry', {})
        cause = 'pass' if r['pass'] else r.get('reason', 'geometry_rejected')
        joined.append({**{k: e.get(k) for k in ('cohort', 'seed', 'name', 'spectral_ratio', 'start', 'target')},
                       **r, 'cause': cause, 'corrected_replay': key in corrected,
                       'max_sin_squared': g.get('max_sin_squared'), 'opposed_flow': g.get('opposed_flow'),
                       'binding_eps': g.get('binding_eps')})
    joined.sort(key=lambda r: (r['job'], r['entry']))
    built = [r for r in joined if 'cold_seconds' in r]
    mismatches = [r for r in joined if r['pass'] != r['python_pass']]
    result = {'entries': len(joined), 'passes': sum(r['pass'] for r in joined),
              'corrective_entries': len(corrected), 'decision_disagreements': len(mismatches),
              'statuses': dict(Counter(r['status'] for r in joined)),
              'remaining': dict(Counter(r['cause'] for r in joined if not r['pass'])),
              'median_cold_seconds': median(r['cold_seconds'] for r in built),
              'median_warm_seconds': median(r['warm_seconds'] for r in built),
              'median_archived_python_speed_ratio': median(r['python_build_seconds']/r['cold_seconds']
                  for r in built if r.get('python_build_seconds')),
              'cohorts': {cohort: {'entries': sum(r['cohort'] == cohort for r in joined),
                          'passes': sum(r['cohort'] == cohort and r['pass'] for r in joined)}
                          for cohort in ('random', 'directed', 'zoo', 'corpus')}}
    repo = Path(__file__).resolve().parents[1]
    result['hash_changes_since_primary'] = [name for name, digest in
        json.loads((args.replay/'manifest.json').read_text())['sha256'].items()
        if hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'entries.json').write_text(json.dumps(joined, indent=2))
    (args.output/'remaining.json').write_text(json.dumps([r for r in joined if not r['pass']], indent=2))
    fields = ['job', 'entry', 'cohort', 'seed', 'name', 'spectral_ratio', 'pass', 'python_pass',
              'cause', 'max_sin_squared', 'opposed_flow', 'binding_eps', 'cold_seconds', 'warm_seconds',
              'corrected_replay']
    for name, subset in [('entries.csv', joined), ('remaining.csv', [r for r in joined if not r['pass']])]:
        with (args.output/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader();writer.writerows(subset)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
