"""WHICH of the funnel's four sign tests refuses, and where.

    python scripts/funnel_probe.py [seed]

farfield_probe established that unstable_endpoint_unresolved is not a reach
problem: on all 9 refusing directed models the branch endpoint is past every
real root of B*N, and the trajectory is EXACTLY the backbone in binary64
(a/a* - 1 = 0.0), so the corridor's width guard is trivially satisfied and all
49 candidate widths are live.

So the refusal is in the four tests themselves.  This replicates the body of
_unstable_far_field_funnel with the results printed per width:

    robust        = (hN)^2 - (B A')^2 w^2        corridor is a real tube
    outward       = -direction * B * N           b is monotone outward
    inward_upper  = -scaled_radial(+w)           upper wall points inward
    inward_lower  =  scaled_radial(-w)           lower wall points inward

Each is checked twice, as the funnel does: cheaply at the endpoint, then
globally on the ray by Sturm.  A test that passes at the endpoint and fails
globally is the expensive kind -- and tells us which polynomial's degree is
worth reducing.
"""
from __future__ import annotations

import os
import random
import sys
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from spong import portrait, topology, _poly as P        # noqa: E402
from qualify import directed_model                       # noqa: E402

NAMES = ("robust", "outward", "inward_upper", "inward_lower")


def probe(m, branch, max_widths: int = 6) -> None:
    endpoint = branch.Y[-1]
    aq, bq = (Fraction.from_float(float(x)) for x in endpoint)
    A, B = m.alpha, m.beta
    Ap, Bp = P.deriv(A), P.deriv(B)
    Aval, Bval = P.eval_at(A, bq), P.eval_at(B, bq)
    print(f"    A(b)={float(Aval):.6g}  B(b)={float(Bval):.6g}")
    if Aval <= 0 or Bval == 0:
        print("    refused early: A <= 0 or B == 0")
        return
    ratio = aq * Aval / Bval - 1
    direction = 1 if float(endpoint[1]) > float(
        branch.diag.get("saddle_b", endpoint[1])) else -1
    hq = Fraction(direction) * bq
    print(f"    direction={direction:+d}  h={float(hq):.6g}  "
          f"ratio={float(ratio):.3e}  h*ratio={float(hq*ratio):.3e}")
    if hq <= 0:
        print("    refused early: h <= 0  (endpoint is on the wrong side)")
        return
    scaled_ratio = hq * ratio

    BAp = P.mul(B, Ap)
    D = P.sub(P.mul(A, Bp), BAp)
    A2 = P.mul(A, A)
    A4 = P.mul(A2, A2)
    h = (Fraction(0), Fraction(direction))
    hN = P.mul(h, m.N)
    outward = P.scale(P.mul(B, m.N), Fraction(-direction))
    AB = P.mul(A, B)
    h3A4 = P.mul(P.mul(P.mul(h, h), h), A4)
    hD = P.mul(h, D)
    hN2 = P.mul(hN, hN)
    BAp2 = P.mul(BAp, BAp)

    def scaled_radial(s):
        h_plus_s = P.add(h, (s,))
        shifted = P.add(hN, P.scale(BAp, s))
        first = P.scale(P.mul(P.mul(h_plus_s, AB), shifted), -direction * s)
        second = P.scale(h3A4, -2 * s)
        third = P.mul(P.mul(P.mul(h_plus_s, h_plus_s), hD), shifted)
        return P.add(P.add(first, second), third)

    print(f"    degrees: outward {P.degree(outward)}  hN^2 {P.degree(hN2)}  "
          f"BAp^2 {P.degree(BAp2)}  A^4 {P.degree(A4)}")
    shown = 0
    for power in range(48, -1, -1):
        width = Fraction(1, 2 ** power)
        if abs(scaled_ratio) >= width or width >= hq:
            continue
        robust = P.sub(hN2, P.scale(BAp2, width * width))
        tests = (robust, outward,
                 P.scale(scaled_radial(width), Fraction(-1)),
                 scaled_radial(-width))
        ends = [P.eval_at(p, bq) > 0 for p in tests]
        line = f"    w=2^-{power:<2d} endpoint " + " ".join(
            f"{n}{'+' if e else '-'}" for n, e in zip(NAMES, ends))
        if not all(ends):
            print(line)
        else:
            rays = [topology._strictly_positive_on_ray(p, bq, direction)
                    for p in tests]
            print(line + "   ray " + " ".join(
                f"{n}{'+' if r else '-'}" for n, r in zip(NAMES, rays)))
            if all(rays):
                print(f"    ACCEPTED at width 2^-{power}")
                return
        shown += 1
        if shown >= max_widths:
            print("    ...")
            return
    print("    no width accepted")


def main() -> int:
    os.environ["SPONG_ENGINE"] = "native"
    os.environ["SPONG_WORKERS"] = "8"
    want = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rng = random.Random(20260806)
    for case in range(20):
        seed = rng.randrange(2 ** 31)
        if want is not None and seed != want:
            continue
        sub = random.Random(seed)
        m, spec = directed_model(sub, 5)
        if m is None:
            continue
        p = portrait.certified_compute(m)
        if p.ledger["topology"].get(
                "resolution_reason") != "unstable_endpoint_unresolved":
            continue
        print(f"\n[{case}] seed {seed}  {spec}")
        for i, br in enumerate(p.branches):
            if br.kind == "unstable" and br.term == "box_exit":
                print(f"  branch {i}")
                probe(m, br)
        if want is not None:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
