"""The observation API preserves legacy results and exports retained vertices."""
import json
from pathlib import Path

import numpy as np
import pytest

from spong import engine
from spong.native_trace import decode


@pytest.mark.parametrize('api,corpus', [
    ('potential_rate_segment','potential_rate.json'),
    ('centered_arrival','centered_arrival.json'),
    ('continue_curve','continue_curve.json'),
])
def test_recording_matches_legacy_calls(monkeypatch,api,corpus):
    from spong import _native
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    import potential_corpus as pc
    import arrival_corpus as ac
    import segment_corpus as sc
    entries=json.loads((root/'tests/corpus'/corpus).read_text())
    original=getattr(_native,api)
    observed=[]
    methods=set()
    trims=[]

    def both(*args):
        assert args[-1] == 1
        recorded=original(*args)
        legacy=original(*args[:-1],0)
        if api=='continue_curve':
            assert recorded[:-1] == legacy[:-1]
            diag=dict(recorded[-1]); blob=diag.pop('trajectory_trace')
            assert diag == legacy[-1]
            points=recorded[-2]
        else:
            # NaN clock fields are deliberately unavailable in both calls.
            for a,b in zip(recorded[:-1],legacy):
                assert a==b or (isinstance(a,float) and np.isnan(a) and np.isnan(b))
            blob=recorded[-1];points=recorded[-2]
        dense,vertices,stats=decode(blob)
        xy=np.frombuffer(points,dtype=float).reshape(-1,2)
        assert len(vertices)==len(xy)
        assert np.array_equal([v[2] for v in vertices.values()],xy)
        methods.update(int(row[21]) for row in dense.values())
        trims.append(stats['trimmed_vertices'])
        observed.append((len(dense),stats))
        return recorded

    monkeypatch.setattr(_native,api,both)
    # Each mode and multiple cases, including arrival finalization/overwrites.
    chosen={}
    for entry in entries:
        chosen.setdefault((entry['case'],entry.get('kind')),entry)
    selected = list(chosen.values())[:8]
    if api == 'continue_curve':
        selected += [e for e in entries if e['output']['term'] == 'enter_shallow']
    for entry in selected:
        m,e,_=sc.context(entry['case'])
        with engine.using('native'):
            if api=='potential_rate_segment':
                pc.run_native(entry['kind'],m,entry['input'])
            elif api=='centered_arrival':
                ac.run_native(m,e,entry['input'])
            else:
                sc.replay(entry)
    assert observed and any(n for n,_ in observed)

    if api == 'continue_curve':
        assert 2 in methods  # actual normalized rescue, not a fabricated tag
        assert any(trims)    # actual native hover-tail trim
