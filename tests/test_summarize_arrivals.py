"""Aggregation must not drop refusals, unprocessed branches or failed jobs."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('summarize_arrivals', Path(__file__).resolve().parents[1]/'scripts/summarize_arrivals.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_censored_entries_and_failed_models_remain_in_denominators(tmp_path):
    manifest = {'specs': [{'cohort': 'random', 'seed': 1}, {'cohort': 'directed', 'seed': 2}], 'sha256': {}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    rows = [
        {'event': 'model', 'status': 'constructed', 'entries': 3},
        {'event': 'entry', 'entry': 4, 'spectral_ratio': 20},
        {'event': 'method', 'entry': 4, 'method': 'raw_arrival', 'pass': True, 'status': 'measured'},
        {'event': 'method', 'entry': 4, 'method': 'analytic', 'pass': False, 'status': 'error', 'reason': 'resonance'},
        {'event': 'entry_refused', 'entry': 5, 'reason': 'target_missing'},
    ]
    (tmp_path/'0000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'0001.jsonl').write_text(json.dumps({'event': 'model', 'status': 'timeout'})+'\n')
    (tmp_path/'completion.jsonl').write_text(json.dumps({'index': 0, 'status': 'worker_timeout'})+'\n')
    result, entries, models = summary.summarize(tmp_path)
    assert result['jobs'] == 2
    assert result['completed'] == 1
    assert result['model_statuses'] == {'constructed': 1, 'timeout': 1}
    assert result['worker_timeouts'] == 1
    assert len(entries) == 3
    assert result['all']['methods']['analytic']['eligible'] == 3
    assert result['all']['methods']['analytic']['pass'] == 0
    assert result['all']['methods']['analytic']['statuses'] == {'error': 1, 'missing': 2}
    assert result['all']['paired']['raw_arrival']['baseline_only'] == 1
    assert result['all']['paired']['raw_arrival']['neither'] == 2
    assert result['entry_refusals'] == {'target_missing': 1, 'unprocessed': 1}


def test_fate_is_separate_and_corpus_has_no_connector_denominator():
    e = {'cohort': 'corpus', 'forced': False, 'methods': {'analytic': {'pass': True, 'status': 'measured'}}}
    result = summary.summarize_entries([e], ('current_connector', 'analytic'))
    assert result['methods']['current_connector']['eligible'] == 0
    assert result['methods']['analytic']['pass'] == 1
    assert result['methods']['analytic']['pass_and_forced'] == 0
    assert sum(result['paired']['current_connector'].values()) == 0


def test_partial_final_line_does_not_erase_previous_events(tmp_path):
    p = tmp_path/'partial.jsonl'
    p.write_text('{"event":"begin"}\n{"eve')
    assert summary.read_rows(p) == [{'event': 'begin'}]
