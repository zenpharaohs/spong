"""Empirical-moment perturbations of the saddle-connection portrait.

The coefficients ``f`` and ``g`` are held at the registered population-wall
member.  Each batch is an independent sample from U(0,1), and its empirical
raw moments define a different exact SPONG model.  The resulting certified
Morse skeletons can be superposed without pretending that a source-labelled
branch has a canonical identity across moment space.

The optional affine-span probe is deliberately only a diagnostic.  It checks
exact Morse inventories at rational points on the segment from the population
moments to the batch moments, but it does not prove that no algebraic wall lies
between two probes.  Consequently the demo never uses nearest critical points
to assert continuation or branch identity.

Usage:
  PYTHONPATH=src:. python3 demos/batch_moment_portraits.py \
      --batch-sizes 32,128,512 --batches 6 --jobs 6
  open out/batch_moment_portraits/nonnearest-saddle-connection-batches.html
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from spong import model, portrait, sturm, zoo


FAMILY = "nonnearest-saddle-connection"


def exact_empirical_moments(samples, count: int):
    """Exact raw moments of binary64 samples.

    ``Fraction.from_float`` treats every generated sample as its exact dyadic
    value.  This removes moment-accumulation roundoff from the experiment; the
    intended perturbation is sampling error alone.
    """
    points = tuple(Fraction.from_float(float(x)) for x in samples)
    n = len(points)
    if n == 0:
        raise ValueError("a batch must contain at least one sample")
    return tuple(
        sum((x**degree for x in points), Fraction(0))/n
        for degree in range(count))


def nested_uniform_batch(seed: int, batch_index: int, size: int):
    """One reproducible prefix of an independent U(0,1) sample stream."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_index]))
    return rng.random(size)


def wall_case():
    return zoo.rheostat_member(FAMILY, "wall")


def build_from_moments(moments):
    case = wall_case()
    return model.build(case.f, case.g, moments)


def morse_signature(enumeration):
    """Ordered endpoint inventory; not a cross-model identity label."""
    return tuple((p.kind, p.source) for p in enumeration.points)


def affine_moments(population, empirical, t: Fraction):
    return tuple((1-t)*x+t*y for x, y in zip(population, empirical))


def probe_affine_span(population, empirical, probes: int):
    """Sample exact Morse inventories on the moment segment.

    This is not an exclusion proof for bifurcations between sample points.
    """
    if probes < 0:
        raise ValueError("span probes must be nonnegative")
    if probes == 0:
        return []
    return [
        {
            "t": float(Fraction(i, probes)),
            "signature": morse_signature(sturm.enumerate_critical_points(
                build_from_moments(affine_moments(
                    population, empirical, Fraction(i, probes))))),
        }
        for i in range(probes+1)
    ]


def _decimate(Y, limit=180):
    Y = np.asarray(Y, dtype=float)
    if len(Y) <= limit:
        return Y
    indices = np.unique(np.linspace(0, len(Y)-1, limit, dtype=int))
    return Y[indices]


def _portrait_payload(p):
    certified = p.ledger["topology"]["status"] == "certified"
    branches = []
    if certified:
        for branch in p.branches:
            Y = _decimate(branch.Y)
            branches.append({
                "kind": branch.kind,
                "points": [[round(float(a), 9), round(float(b), 9)]
                           for a, b in Y],
            })
    return {
        "certified": certified,
        "topology_status": p.ledger["topology"]["status"],
        "resolution_reason": p.ledger["topology"]["resolution_reason"],
        "morse_signature": morse_signature(p.enumeration),
        "critical_points": [
            {
                "a": round(float(q.a), 12),
                "b": round(float(q.b), 12),
                "kind": q.kind,
                "source": q.source,
                "loss": float(p.model.L(q.a, q.b)),
            }
            for q in p.enumeration.points
        ],
        "branches": branches,
        "timing_sec": float(p.ledger["timing"]["total_sec"]),
    }


def compute_batch(task):
    size, batch_index, seed, max_geometry_level, span_probes = task
    case = wall_case()
    count = 2*max(len(case.f)-1, len(case.g)-1)+1
    population = model.moments_uniform01(count)
    samples = nested_uniform_batch(seed, batch_index, size)
    empirical = exact_empirical_moments(samples, count)
    m = build_from_moments(empirical)
    family = zoo.get_wall_family(FAMILY)
    p = portrait.certified_compute(
        m, view=family.default_view,
        max_geometry_level=max_geometry_level)
    payload = _portrait_payload(p)
    payload.update({
        "batch_size": size,
        "batch_index": batch_index,
        "seed": seed,
        "moment_error_l2": float(np.linalg.norm([
            float(x-y) for x, y in zip(empirical, population)])),
        "moment_error_max": max(
            abs(float(x-y)) for x, y in zip(empirical, population)),
        "moments": [float(x) for x in empirical],
        "span_probe": probe_affine_span(
            population, empirical, span_probes),
        "span_probe_is_proof": False,
    })
    return payload


def _html(report):
    data = json.dumps(report, separators=(",", ":"))
    return f"""<!doctype html>
<meta charset="utf-8"><title>SPONG empirical-batch portraits</title>
<style>
body{{font-family:system-ui;margin:24px;background:#fafafa;color:#222}}
main{{max-width:1200px;margin:auto}} .controls{{display:flex;gap:14px;flex-wrap:wrap}}
svg{{width:100%;height:auto;background:white;border:1px solid #bbb}}
button{{padding:6px 12px}} .note{{max-width:80ch}} code{{background:#eee}}
</style>
<main><h1>Empirical-batch perturbations at the population handle slide</h1>
<p class="note">Each color is one nested U(0,1) batch stream. Solid curves
are unstable manifolds; dashed curves are stable manifolds. Only portraits
whose global topology audit certified are drawn. No branch identity is
asserted across moment space.</p>
<div class="controls" id="sizes"></div>
<svg id="plot" viewBox="0 0 1120 610" role="img"
 aria-label="Superposed certified empirical-batch Morse skeletons"></svg>
<p id="status"></p></main>
<script>
const report={data};
const colors=['#386cb0','#f0027f','#7fc97f','#fdc086','#beaed4','#bf5b17'];
const view=report.view, svg=document.getElementById('plot');
const sx=a=>54+(a-view[0])*1012/(view[1]-view[0]);
const sy=b=>558-(b-view[2])*518/(view[3]-view[2]);
const path=pts=>'M'+pts.map(p=>sx(p[0]).toFixed(2)+','+sy(p[1]).toFixed(2)).join('L');
function el(name,attrs={{}}){{const x=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,v);return x}}
function draw(size){{
 svg.replaceChildren();
 svg.append(el('rect',{{x:54,y:40,width:1012,height:518,fill:'white',stroke:'#aaa'}}));
 const group=report.results.filter(x=>x.batch_size===size);
 group.forEach((item,i)=>{{if(!item.certified)return;item.branches.forEach(br=>{{
   const p=el('path',{{d:path(br.points),fill:'none',stroke:colors[i%colors.length],
     'stroke-width':br.kind==='unstable'?2.0:1.55,
     'stroke-dasharray':br.kind==='stable'?'6 4':'none','stroke-opacity':.72}});svg.append(p);
 }});item.critical_points.forEach(q=>{{const c=el('circle',{{cx:sx(q.a),cy:sy(q.b),r:q.kind==='min'?3.3:2.6,fill:colors[i%colors.length],stroke:'white','stroke-width':.8}});svg.append(c)}})}});
 const certified=group.filter(x=>x.certified).length;
 document.getElementById('status').textContent=`N=${{size}}: ${{certified}}/${{group.length}} certified; median max moment error ${{median(group.map(x=>x.moment_error_max)).toExponential(2)}}.`;
 }}
function median(x){{x=x.slice().sort((a,b)=>a-b);return x[Math.floor(x.length/2)]}}
const sizes=[...new Set(report.results.map(x=>x.batch_size))];
sizes.forEach((n,i)=>{{const b=document.createElement('button');b.textContent='N='+n;b.onclick=()=>draw(n);document.getElementById('sizes').append(b)}});
draw(sizes[Math.min(1,sizes.length-1)]);
</script>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Superpose certified portraits from empirical moments")
    parser.add_argument("--batch-sizes", default="32,128,512")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-geometry-level", type=int, default=1)
    parser.add_argument("--span-probes", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("out/batch_moment_portraits"))
    parser.add_argument(
        "--merge", nargs="+", type=Path,
        help="merge worker JSON reports instead of computing portraits")
    args = parser.parse_args(argv)
    family = zoo.get_wall_family(FAMILY)
    if args.merge is not None:
        results = []
        for path in args.merge:
            results.extend(json.loads(path.read_text())["results"])
        results.sort(key=lambda x: (x["batch_size"], x["batch_index"]))
        report = {
            "format": "spong-batch-moment-portraits-v1",
            "family": FAMILY,
            "distribution": "uniform01",
            "fixed_coefficients": "population-wall f and g",
            "view": family.default_view,
            "branch_identity_across_batches": False,
            "span_probe_is_proof": False,
            "results": results,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{FAMILY}-batches"
        (args.output_dir/f"{stem}.json").write_text(
            json.dumps(report, indent=2)+"\n")
        (args.output_dir/f"{stem}.html").write_text(_html(report))
        print(args.output_dir/f"{stem}.html")
        return report
    sizes = tuple(int(x) for x in args.batch_sizes.split(",") if x)
    if (not sizes or any(x <= 0 for x in sizes) or args.batches <= 0
            or args.jobs <= 0):
        parser.error("batch sizes, batch count, and jobs must be positive")
    tasks = [
        (size, index, args.seed, args.max_geometry_level, args.span_probes)
        for size in sizes
        for index in range(args.batch_offset,
                           args.batch_offset+args.batches)]
    if args.jobs == 1:
        results = list(map(compute_batch, tasks))
    else:
        # Some restricted installation validators forbid POSIX semaphore
        # queries even though threads are available.  The production C core
        # releases the GIL during its expensive exact and continuation work,
        # so a thread fallback still provides useful parallelism there.
        try:
            pool_type = ProcessPoolExecutor
            pool = pool_type(max_workers=args.jobs)
        except PermissionError:
            pool_type = ThreadPoolExecutor
            pool = pool_type(max_workers=args.jobs)
        with pool:
            results = list(pool.map(compute_batch, tasks))
    results.sort(key=lambda x: (x["batch_size"], x["batch_index"]))
    report = {
        "format": "spong-batch-moment-portraits-v1",
        "family": FAMILY,
        "distribution": "uniform01",
        "fixed_coefficients": "population-wall f and g",
        "view": family.default_view,
        "branch_identity_across_batches": False,
        "span_probe_is_proof": False,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{FAMILY}-batches"
    (args.output_dir/f"{stem}.json").write_text(
        json.dumps(report, indent=2)+"\n")
    (args.output_dir/f"{stem}.html").write_text(_html(report))
    print(args.output_dir/f"{stem}.html")
    return report


if __name__ == "__main__":
    main()
