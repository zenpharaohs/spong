#!/usr/bin/env python3
"""Experimental native precision-bridge replay at frozen entry points."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def worker(source, job, evidence_dir, check_precision):
    os.environ['SPONG_ENGINE'] = 'native'
    os.environ['SPONG_WORKERS'] = '1'
    os.environ['SPONG_STUB_MODE'] = 'jet'
    import numpy as np
    from spong import model, sturm, zoo
    from spong.precision_bridge import checked_precision_path as checked_path
    from spong.trajectory_check import inventory
    from qualify_arrivals import deadline
    import qualify
    entries = [e for e in json.loads(source.read_text()) if e['job'] == job]
    spec = entries[0]
    emit({'event': 'begin', 'job': job, 'entries': len(entries)})
    with deadline(90):
        if spec['cohort'] in ('random', 'directed'):
            generate = qualify.random_model if spec['cohort'] == 'random' else qualify.directed_model
            m, _ = generate(random.Random(spec['seed']), spec.get('max_degree', 5))
        else:
            z = zoo.get(spec['name'])
            moments = model.moments_normal01 if z.moment_dist == 'normal01' else model.moments_uniform01
            m = model.build(list(z.f), list(z.g), moments(2*max(len(z.f), len(z.g))-1))
        en = sturm.materialize_stubs(m, sturm.enumerate_critical_points(m))
    for e in entries:
        t = time.perf_counter()
        row = {'event': 'arrival', 'job': job, 'entry': e['entry'], 'cohort': e['cohort']}
        try:
            with deadline(240):
                q = next(q for q in en.minima if [q.a, q.b] == e['target'])
                points, pieces, checks = checked_path(m, q, e['start'], e['target'],
                    sample_budget=50000, check_precision=check_precision)
                coverage = inventory(points, pieces)
                assert coverage['status'] == 'complete'
                assert np.array_equal(points[0], e['start']) and np.array_equal(points[-1], e['target'])
                row.update(status='measured', **{'pass': True}, checks=checks,
                           coverage=coverage, pieces=[p.summary() for p in pieces],
                           handoff=len(pieces)>1, matching=pieces[-1].evidence.get('matching'))
                if len(pieces)>1 and evidence_dir:
                    packed = []
                    for p in pieces:
                        ev = dict(p.evidence)
                        if isinstance(ev.get('trace'), bytes):
                            ev['trace_base64'] = base64.b64encode(ev.pop('trace')).decode('ascii')
                        packed.append({'kind': p.kind, 'start': p.start, 'stop': p.stop, 'evidence': ev})
                    (evidence_dir/f'{job:04d}-{e["entry"]:04d}.json').write_text(
                        json.dumps({'job': job, 'entry': e['entry'], 'points': points.tolist(),
                                    'pieces': packed, 'checks': checks}, allow_nan=False))
        except Exception as exc:
            row.update(status='timeout' if isinstance(exc, TimeoutError) else 'error',
                       **{'pass': False}, reason=f'{type(exc).__name__}: {exc}',
                       handoff_diagnostics=getattr(exc, 'handoff', None))
        row['seconds'] = time.perf_counter()-t
        emit(row)
    emit({'event': 'end', 'job': job})


def run_one(args, job):
    with (args.output/f'{job:04d}.jsonl').open('w') as log, (args.output/f'{job:04d}.stderr').open('w') as err:
        try:
            p = subprocess.run([sys.executable, __file__, str(args.source), '--worker', str(job),
                '--output', str(args.output), '--check-precision', str(args.check_precision)],
                cwd=REPO, stdout=log, stderr=err, timeout=900,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            return {'job': job, 'returncode': p.returncode}
        except subprocess.TimeoutExpired:
            return {'job': job, 'status': 'worker_timeout'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--worker', type=int)
    ap.add_argument('--check-precision', type=int, default=90)
    args = ap.parse_args()
    args.source=args.source.resolve();args.output=args.output.resolve()
    if args.worker is not None:
        worker(args.source, args.worker, args.output/'evidence', args.check_precision)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'evidence').mkdir()
    entries = json.loads(args.source.read_text())
    jobs = sorted({r['job'] for r in entries})
    paths = [*sorted((REPO/'src/spong').glob('*.py')), *sorted((REPO/'src/spong').glob('_native*.so')),
             *sorted((REPO/'src/c').glob('*.c')), *sorted((REPO/'src/spong').glob('_native*.inc')),
             *sorted((REPO/'include/spong').glob('*.h')), Path(__file__).resolve()]
    manifest = {'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'entries': len(entries), 'jobs': jobs, 'workers': args.workers, 'check_precision': args.check_precision,
        'sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (args.output/'completion.jsonl').open('w') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, args, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result();log.write(json.dumps(r)+'\n');log.flush()
            print(f'{n}/{len(jobs)} completed: {r}', flush=True)


if __name__ == '__main__':
    main()
