"""Independent analytic controls and adversarial native evidence."""
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from spong.native_trace import decode, measure_native, physical, polynomial
from spong.trajectory import TrajectoryPiece as Piece


def hermite(a, b, f, g):
    row = np.zeros(24)
    row[0] = 2
    row[3] = 1
    row[4:12] = (*a, *b, *f, *g)
    return row


def vertex(i, point, source, t=1):
    row = np.zeros(24)
    row[0] = 3
    row[4:6] = point
    row[20], row[22], row[23] = i, t, source
    return row


def proposal(rows, points):
    return np.array(points, dtype=float), Piece("potential_ascent", 0, len(points)-1, {
        "representation": "native_dense_trace", "schema": 1,
        "trace": np.array(rows).tobytes(), "source_points": len(points),
        "branch_offset": 0})


MODEL = SimpleNamespace(alpha=(Q(1),), beta=(Q(0),))


def test_hermite_true_tangent_not_chord_or_producer_gradient():
    # a=1+t, b=t²: tangent (1,2t) against grad L=(2a,0).
    row = hermite((1,0), (2,1), (1,0), (1,2))
    pts, piece = proposal([vertex(0,(1,0),-1), row,
                           vertex(1,(2,1),1)], [(1,0),(2,1)])
    r = measure_native(MODEL, pts, piece, 1)
    assert r['status'] == 'measured'
    assert r['max_sin_squared'] == pytest.approx(4/5)
    assert r['binding_eps'] == r['join_eps'] == 0
    assert measure_native(MODEL, pts, piece, -1)['opposed_flow'] == 5
    assert measure_native(MODEL, pts, piece, 1, sample_budget=4)['reason'] == 'measurement_budget'
    pts[1,0] += 0.01
    assert measure_native(MODEL, pts, piece, 1)['reason'] == 'geometry_mismatch'


@pytest.mark.parametrize('chart', [0, 1, 2, 3])
def test_collocation_reconstruction_and_coordinate_pullback(chart):
    # Constant stage derivatives integrate exactly, independent of tracer code.
    row = np.zeros(24)
    row[0:4] = (1,chart,4,0.5)
    row[4:10] = (1,2,3,4,3,4)
    row[14:16] = (0.25,0.75)
    row[22:24] = (8,16)
    c = polynomial(row)
    assert c == ((Q(1),Q(3,2),0),(Q(2),Q(2),0))
    p,v = physical(MODEL,row,c,Q(1))
    expected = {0: ((Q(5,2),4),(Q(3,2),2)),
                1: ((4,Q(5,2)),(2,Q(3,2))),
                2: ((Q(5,2),4),(Q(3,2),2)),
                3: ((Q(21,2),20),(Q(3,2),2))}
    assert (p,v) == expected[chart]


def test_trim_discards_bad_tail_and_replacement_preserves_mapping():
    good = hermite((1,0),(2,0),(1,0),(1,0))
    bad = hermite((2,0),(1,1),(-1,1),(-1,1))
    trim = np.zeros(24); trim[0]=4; trim[20]=2
    rows = [vertex(0,(1,0),-1), good, vertex(1,(2,0),1),
            bad,vertex(2,(1,1),3),trim,vertex(1,(2,0),1)]
    pts,piece = proposal(rows,[(1,0),(2,0)])
    r = measure_native(MODEL,pts,piece,1)
    assert r['status'] == 'measured'
    assert r['steps'] == r['discarded_steps'] == r['trimmed_vertices'] == r['replaced_vertices'] == 1
    assert r['max_sin_squared'] == r['opposed_flow'] == 0


@pytest.mark.parametrize('mutation,reason', [
    (lambda r: r[1].__setitem__(21,9), 'unknown_dense_representation'),
    (lambda r: r[2].__setitem__(23,900), 'missing_dense_source'),
    (lambda r: r[2].__setitem__(22,2), 'invalid_parameter'),
    (lambda r: r[2].__setitem__(20,3), 'incomplete_vertex_map'),
    (lambda r: r[1].__setitem__(8,np.nan), 'nonfinite_trace'),
    (lambda r: r[2].__setitem__(23,-1), 'unrepresented_vertex'),
])
def test_malformed_and_unrepresented_evidence_refused(mutation,reason):
    rows = [vertex(0,(1,0),-1),hermite((1,0),(2,0),(1,0),(1,0)),vertex(1,(2,0),1)]
    mutation(rows)
    pts,piece = proposal(rows,[(1,0),(2,0)])
    assert measure_native(MODEL,pts,piece,1)['reason'] == reason


def test_partial_dense_endpoint_and_unvalidated_rescue():
    row = hermite((1,0),(2,0),(1,0),(1,0)); row[21]=3
    pts,piece = proposal([vertex(0,(1,0),-1),row,vertex(1,(1.5,0),1,0.5)],[(1,0),(1.5,0)])
    r = measure_native(MODEL,pts,piece,1)
    assert r['samples'] == 3
    assert r['unvalidated_steps'] == 1
    assert r['binding_eps'] == 0
    assert measure_native(MODEL,pts,replace(piece,evidence={**piece.evidence,'schema':2}),1)['reason'] == 'unknown_trace_schema'
    with pytest.raises(ValueError,match='malformed_trace'):
        decode(b'bad')


def test_first_dense_start_must_bind_to_incoming_anchor():
    row = hermite((10,0),(11,0),(1,0),(1,0))
    points,piece = proposal([vertex(0,(1,0),-1),row,vertex(1,(11,0),1)],
                            [(1,0),(11,0)])
    result = measure_native(MODEL,points,piece,1)
    assert result['max_sin_squared'] == 0
    assert result['binding_eps'] == 0
    assert result['join_eps'] > 64
