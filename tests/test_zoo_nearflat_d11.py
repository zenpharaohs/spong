"""The stiff Gaussian geometry example remains reproducible as a zoo preset."""

from demos.explorer import serve
from spong import model, sturm, zoo


def test_normal_symmetric_nearflat_d11_preset():
    name = "normal-symmetric-nearflat-d11"
    assert name in zoo.names()
    case = zoo.get(name)
    expected_f = (0, -3) * 6
    expected_g = (-3, 1, -3, -10, -3, 31, -3, -35, -3, 11, -3, -1)
    assert case.f == expected_f
    assert case.g == expected_g
    assert case.moment_dist == "normal01"

    f, g, view, spec, _ = serve._resolve({"zoo": name})
    assert tuple(f) == expected_f
    assert tuple(g) == expected_g
    assert view is None
    assert spec == {"kind": "normal01"}

    moments = zoo.moments_for(case, 23)
    assert moments == model.moments_normal01(23)
    enumeration = sturm.enumerate_critical_points(model.build(f, g, moments))
    assert enumeration.morse and enumeration.psi_positive
    assert (len(enumeration.minima), len(enumeration.saddles)) == (4, 5)
