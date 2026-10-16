"""A failed chart step must not become a drawn separatrix corner."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spong import atlas, charts, engine, model, portrait, sturm
from demos.explorer import serve


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import segment_corpus  # noqa: E402


@pytest.mark.parametrize("index", [8, 17, 23])
def test_stiff_continuation_keeps_accepted_turns_resolved(index):
    # These examples previously accepted an off-flow step or a failed
    # candidate at the retry floor, including an almost 180-degree reversal.
    entry = json.loads((ROOT / "tests/corpus/continue_curve.json").read_text())[index]
    m, enumeration, _ = segment_corpus.context(entry["case"])
    inp = entry["input"]
    local = segment_corpus.find_local(
        enumeration, *inp["centered_local_at"])
    curves = []
    for selected in ("python", "native"):
        with engine.using(selected):
            points, _term, _switches, _end = charts._continue_curve(
                m, inp["b"], inp["w"], inp["flow"],
                [tuple(t) for t in inp["targets"]], tuple(inp["box"]),
                inp["ds"], max_steps=1000, ds0=inp["ds0"],
                centered_local=local,
                **({} if inp["cap_r"] is None else
                   {"cap_r": inp["cap_r"]}))
        curves.append(np.asarray(points))
        chords = np.diff(curves[-1], axis=0)
        lengths = np.hypot(chords[:, 0], chords[:, 1])
        assert np.all(lengths <= 2.0*inp["ds"])
        for j, (a, b) in enumerate(curves[-1][:-1]):
            grad = m.gradL(a, b)
            norm = float(np.hypot(*grad))
            scale_a = 2.0*(abs(a)*m.A(b)+abs(m.B(b)))
            scale_b = 2.0*abs(a)*abs(m.Bp(b))+a*a*abs(m.Ap(b))
            floor = 16.0*np.finfo(float).eps * float(
                np.hypot(scale_a, scale_b))
            if norm >= max(1e-12, 1e3*floor) and lengths[j] > 0.0:
                alignment = -inp["flow"]*float(grad @ chords[j]) / (
                    norm*lengths[j])
                assert alignment >= charts.FIELD_ALIGNMENT_MIN
        for j in range(1, len(chords)):
            a, b = curves[-1][j+1]
            w = a - m.s_a_star(b)
            resolution = charts.TURN_RES_FACTOR*np.finfo(float).eps * (
                abs(m.s_a_star(b)) + abs(w) + abs(b))
            if lengths[j-1] <= resolution or lengths[j] <= resolution:
                continue
            cosine = float(chords[j-1] @ chords[j]) / (
                lengths[j-1]*lengths[j])
            assert cosine >= charts.TURN_MAX
    assert np.array_equal(*curves)


@pytest.fixture(scope="module")
def degree11_case():
    pytest.importorskip("spong._native")
    m = model.build(
        [0, -3]*6,
        [-3, 1, -3, -10, -3, 31, -3, -35, -3, 11, -3, -1],
        model.moments_normal01(23))
    enumeration = sturm.materialize_stubs(
        m, sturm.enumerate_critical_points(m))
    bmax = atlas.legal_max_b(m)
    amax = bmax/math.sqrt(atlas.effective_degree(m))
    display = atlas.compute_box(
        m, enumeration, view=(-amax, amax, -bmax, bmax))
    ds = max(display[1]-display[0], display[3]-display[2])/30000.0
    stop_loss = (max(float(m.L(float(m.a_star(q.b)), q.b))
                     for q in enumeration.saddles)*(1.0+1e-9)+1e-12)
    box = portrait._stable_trace_box(m, display, stop_loss)
    critical = [(float(q.a), float(q.b)) for q in enumeration.points]
    return m, enumeration, box, ds, critical, stop_loss


@pytest.mark.parametrize("saddle_b,sign", [(-0.14, -1), (0.14, 1)])
def test_gaussian_degree11_stable_branch_refuses_false_box_exit(
        degree11_case, saddle_b, sign):
    m, enumeration, box, ds, critical, stop_loss = degree11_case
    saddle = min(enumeration.saddles,
                 key=lambda q: abs(float(q.b)-saddle_b))
    with engine.using("native"):
        branch = charts.trace_stable(
            m, saddle.b, sign, box=box, ds=ds,
            critical_local=saddle.local,
            critical_stub=portrait._stable_stub(saddle, sign, m),
            critical_points=critical, stop_loss=stop_loss)
    lengths = np.linalg.norm(np.diff(branch.Y, axis=0), axis=1)
    assert branch.term == "abort_step_failure"
    assert np.max(lengths) <= 8.0*ds
    assert branch.certs["angle_energy"] < 1e-6


def test_gaussian_degree11_growth_refuses_off_flow_join(degree11_case):
    m, enumeration, box, ds, critical, stop_loss = degree11_case
    saddle = min(enumeration.saddles, key=lambda q: float(q.b))
    with engine.using("native"):
        branch = charts.trace_stable(
            m, saddle.b, +1, box=box, ds=ds,
            critical_local=saddle.local,
            critical_stub=portrait._stable_stub(saddle, +1, m),
            critical_points=critical, stop_loss=stop_loss)
        tail = serve._j_tails(
            m, SimpleNamespace(branches=[branch], box=box),
            float(atlas.effective_degree(m)),
            enumeration=enumeration)[0]
    assert tail["grow_term"] in (
        "abort_step_failure", "growth_geometry_failure")
    assert tail["sector"] is None
    assert tail["n_grown"] < 100
