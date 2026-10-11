"""Named ripple presets retain their own laws across zoo entry points."""

import pytest

from demos.explorer import serve
from spong import cli, comparison_cli, portrait, sturm, zoo


@pytest.mark.parametrize("name,law,expected_mu4", [
    ("normal-linear-ripple-d5", "normal01", 3),
    ("rademacher-chebyshev-d5", "rademacher", 1),
])
def test_named_ripple_presets_route_moments_and_certify(name, law, expected_mu4):
    case = zoo.get(name)
    assert case.moment_dist == law
    f, g, _, spec, _ = serve._resolve({"zoo": name})
    assert tuple(f) == case.f and tuple(g) == case.g
    assert spec == {"kind": law}
    moments = zoo.moments_for(case, 11)
    assert moments[4] == expected_mu4
    assert cli._moments(law, 11) == moments
    assert serve.moment_vector(spec, 11) == moments
    m = comparison_cli._zoo_model(case)
    assert m.mu == moments
    e = sturm.enumerate_critical_points(m)
    assert e.psi_positive and e.morse
    assert (len(e.saddles), len(e.minima)) == (5, 4)
    p = portrait.compute(m)
    assert p.ledger["summary"]["topology_status"] == "certified"
    assert p.ledger["summary"]["representation_attested"]


def test_zoo_moment_law_rejects_unknown_distribution():
    case = zoo.ZooCase("invalid-law", (0, 1), (1, 1), "unknown", "")
    with pytest.raises(ValueError, match="unsupported zoo moment law"):
        zoo.moments_for(case, 5)
