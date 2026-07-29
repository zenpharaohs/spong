"""Command-line inspection tools for random phase portraits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import model, portrait, render, zoo


def _coefficients(rng: np.random.Generator, degree: int,
                  dist: str, scale: float) -> list[float]:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if dist == "normal":
        c = rng.normal(0.0, scale, degree + 1)
    elif dist == "uniform":
        c = rng.uniform(-scale, scale, degree + 1)
    else:
        raise ValueError(f"unknown coefficient distribution {dist!r}")
    if degree > 0 and abs(c[-1]) < 1e-12 * max(1.0, scale):
        c[-1] = np.copysign(1e-12 * max(1.0, scale), c[-1] or 1.0)
    return [float(x) for x in c]


def _moments(name: str, n: int):
    if name == "uniform01":
        return model.moments_uniform01(n)
    if name == "normal01":
        return model.moments_normal01(n)
    raise ValueError(f"unknown moment distribution {name!r}")


def _view(values: list[float] | None):
    if values is None:
        return None
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--view needs four floats")
    return tuple(float(x) for x in values)


def _branch_summary(p: portrait.Portrait) -> list[dict]:
    out = []
    for br in p.branches:
        out.append({
            "kind": br.kind,
            "term": br.term,
            "n_points": int(len(br.Y)),
            "saddle_b": br.diag.get("saddle_b"),
            "switches": br.diag.get("switches"),
            "angle_energy": br.certs.get("angle_energy"),
            "seam_residual": br.certs.get("seam_residual"),
        })
    return out


def _inkscape_command() -> list[str] | None:
    exe = shutil.which("inkscape")
    if exe:
        return [exe]
    for app in ("/Applications/Inkscape.app",
                str(Path.home() / "Applications" / "Inkscape.app")):
        if Path(app).exists():
            return ["open", "-a", "Inkscape"]
    return None


def _browser_command() -> list[str] | None:
    for exe in ("msedge", "chrome", "firefox"):
        found = shutil.which(exe)
        if found:
            return [found]
    if os.name == "nt":
        for candidate in (
            Path(os.environ.get("ProgramFiles(x86)", "")) /
            "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.environ.get("ProgramFiles", "")) /
            "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.environ.get("LocalAppData", "")) /
            "Microsoft" / "Edge" / "Application" / "msedge.exe",
        ):
            if candidate.exists():
                return [str(candidate)]
    return None


def _open_one(path: str, viewer: str = "auto") -> bool:
    inkscape = _inkscape_command()
    if viewer == "inkscape":
        if inkscape is None:
            return False
        cmd = [*inkscape, path]
    elif viewer == "browser" or (viewer == "auto" and os.name == "nt"):
        browser = _browser_command()
        if browser is None:
            if viewer == "browser":
                return False
            return _open_one(path, viewer="open")
        cmd = [*browser, Path(path).resolve().as_uri()]
    elif viewer == "open" or viewer == "auto":
        if sys.platform == "darwin":
            cmd = ["open", path]
        elif os.name == "nt":
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except OSError:
                return False
            return True
        else:
            cmd = ["xdg-open", path]
    else:
        raise ValueError(f"unknown viewer {viewer!r}")
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return True


def _open_outputs(paths: list[str | None], viewer: str) -> int:
    opened = 0
    for path in paths:
        if path is None:
            continue
        if _open_one(str(Path(path).resolve()), viewer=viewer):
            opened += 1
    return opened


def _case_summary(case_seed: int, f, g, p: portrait.Portrait,
                  elapsed: float, plane_path: str | None,
                  disk_path: str | None, zooms: list[dict],
                  render_view=None) -> dict:
    e = p.enumeration
    return {
        "seed": case_seed,
        "f": f,
        "g": g,
        "box": tuple(float(x) for x in p.box),
        "view": None if p.view is None else tuple(float(x) for x in p.view),
        "render_view": None if render_view is None
        else tuple(float(x) for x in render_view),
        "elapsed_sec": elapsed,
        "plane_svg": plane_path,
        "disk_svg": disk_path,
        "zooms": zooms,
        "enumeration": {
            "n_critical": len(e.points),
            "n_min": len(e.minima),
            "n_saddle": len(e.saddles),
            "psi_positive": e.psi_positive,
            "morse": e.morse,
            "alternates": e.alternates,
        },
        "ledger_summary": p.ledger.get("summary", {}),
        "branches": _branch_summary(p),
    }


def _render_plane_path(p: portrait.Portrait, path: str, view, args,
                       title: str) -> str:
    svg = render.plane_view(
        p, view=view, width=args.width, height=args.height,
        n_levels=args.levels, n_grid=args.grid, title=title)
    return render.save(svg, path)


def _parse_view_text(text: str):
    parts = text.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("expected four numbers: A_LO A_HI B_LO B_HI")
    view = tuple(float(x) for x in parts)
    if not (view[0] < view[1] and view[2] < view[3]):
        raise ValueError("expected A_LO < A_HI and B_LO < B_HI")
    return view


def _view_inside_box(view, box) -> bool:
    return (box[0] <= view[0] and view[1] <= box[1]
            and box[2] <= view[2] and view[3] <= box[3])


def _fit_view_to_box(view, box):
    if view is None:
        return None
    fitted = (max(view[0], box[0]), min(view[1], box[1]),
              max(view[2], box[2]), min(view[3], box[3]))
    if fitted[0] >= fitted[1] or fitted[2] >= fitted[3]:
        return tuple(float(x) for x in box)
    return fitted


def _maybe_report_view_clamp(requested, fitted):
    if requested is not None and tuple(requested) != tuple(fitted):
        print(f"  requested view exceeds compute box; rendering {fitted}")


def _pause_loop(p: portrait.Portrait, out_dir: Path, stem: str,
                title: str, args) -> str:
    n_view = 0
    while True:
        reply = input(
            "Enter=next, v=new view box, q=stop: ").strip().lower()
        if reply.startswith("q"):
            return "quit"
        if not reply:
            return "next"
        if reply.startswith("v"):
            raw = input("view A_LO A_HI B_LO B_HI: ").strip()
            try:
                view = _parse_view_text(raw)
            except ValueError as exc:
                print(f"  invalid view: {exc}")
                continue
            if not _view_inside_box(view, p.box):
                print("  recomputing portrait for new compute box...")
                t0 = time.perf_counter()
                p = portrait.compute(
                    p.model, view=view,
                    trace_stable_branches=not args.no_stable)
                print(f"  new compute box: {tuple(float(x) for x in p.box)} "
                      f"({time.perf_counter() - t0:.2f}s)")
            render_view = _fit_view_to_box(view, p.box)
            _maybe_report_view_clamp(view, render_view)
            path = str(out_dir / f"{stem}_view_{n_view:02d}.svg")
            n_view += 1
            _render_plane_path(
                p, path, render_view, args,
                f"{title} | view {render_view[0]:.4g} {render_view[1]:.4g} "
                f"{render_view[2]:.4g} {render_view[3]:.4g}")
            print(f"  view:  {path}")
            if args.auto_open:
                if _open_outputs([path], viewer=args.viewer):
                    print(f"  opened view with viewer={args.viewer}")
            continue
        print("  commands: Enter for next, v for new view, q to stop")


def random_phase_portrait(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summaries = []
    for idx in range(args.count):
        case_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        case_rng = np.random.default_rng(case_seed)
        f = _coefficients(case_rng, args.f_degree, args.coeff_dist,
                          args.coeff_scale)
        if args.same:
            g = list(f)
        else:
            g = _coefficients(case_rng, args.g_degree, args.coeff_dist,
                              args.coeff_scale)
        moments = _moments(args.moment_dist,
                           2 * max(args.f_degree, args.g_degree) + 1)
        title = (f"random portrait seed={case_seed} "
                 f"deg(f)={args.f_degree} deg(g)={args.g_degree}"
                 f"{' f=g' if args.same else ''}")
        t0 = time.perf_counter()
        m = model.build(f, g, moments)
        p = portrait.compute(
            m, view=args.view,
            trace_stable_branches=not args.no_stable)
        elapsed = time.perf_counter() - t0

        stem = f"{args.prefix}_{idx:04d}_seed_{case_seed}"
        plane_path = disk_path = None
        render_view = _fit_view_to_box(args.view, p.box)
        _maybe_report_view_clamp(args.view, render_view)
        if args.view_kind in ("plane", "both"):
            plane_path = str(out_dir / f"{stem}_plane.svg")
            _render_plane_path(p, plane_path, render_view, args, title)
        if args.view_kind in ("disk", "both"):
            disk_path = str(out_dir / f"{stem}_disk.svg")
            svg = render.disk_view(
                p, width=args.disk_size, height=args.disk_size,
                n_levels=max(1, args.levels // 2), title=title)
            render.save(svg, disk_path)

        zoom_paths = []
        zooms = []
        if args.zoom_close > 0:
            for zi, z in enumerate(render.close_unstable_zooms(
                    p, n=args.zoom_close, samples=args.zoom_samples)):
                ztitle = (f"{title} | zoom {zi}: "
                          f"unstable branches {z['branches'][0]},"
                          f"{z['branches'][1]} sep={z['separation']:.3e}")
                zpath = str(out_dir / f"{stem}_zoom_{zi:02d}.svg")
                svg = render.plane_view(
                    p, view=z["view"], width=args.width,
                    height=args.height, n_levels=args.zoom_levels,
                    n_grid=args.zoom_grid, title=ztitle)
                render.save(svg, zpath)
                zoom_paths.append(zpath)
                zooms.append({**z, "svg": zpath})

        summary = _case_summary(case_seed, f, g, p, elapsed,
                                plane_path, disk_path, zooms, render_view)
        summaries.append(summary)
        one_path = out_dir / f"{stem}_summary.json"
        one_path.write_text(json.dumps(summary, indent=2) + "\n")

        led = summary["ledger_summary"]
        print(
            f"[{idx + 1}/{args.count}] seed={case_seed} "
            f"crit={summary['enumeration']['n_critical']} "
            f"branches={len(summary['branches'])} "
            f"Emax={led.get('worst_angle_energy', 0.0):.3e} "
            f"turn={led.get('worst_max_turn_deg', 0.0):.3f} "
            f"balanced={led.get('balanced')} "
            f"time={elapsed:.2f}s")
        if plane_path:
            print(f"  plane: {plane_path}")
        if disk_path:
            print(f"  disk:  {disk_path}")
        if args.no_stable:
            print("  stable separatrices skipped (--no-stable)")
        for zi, z in enumerate(zooms):
            print(f"  zoom{zi}: {z['svg']} "
                  f"sep={z['separation']:.3e} view={z['view']}")
        print(f"  json:  {one_path}")

        if args.auto_open:
            opened = _open_outputs([plane_path, disk_path, *zoom_paths],
                                   viewer=args.viewer)
            if opened:
                print(f"  opened {opened} SVG view{'s' if opened != 1 else ''}"
                      f" with viewer={args.viewer}")
            else:
                print(f"  viewer={args.viewer} did not open an SVG")

        if args.pause:
            action = _pause_loop(p, out_dir, stem, title, args)
            if action == "quit":
                break

    index_path = out_dir / f"{args.prefix}_index.json"
    index_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"index: {index_path}")
    return 0


def zoo_phase_portrait(args: argparse.Namespace) -> int:
    z = zoo.get(args.zoo)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view = args.view if args.view is not None else z.default_view
    moments = _moments(z.moment_dist, 2 * max(len(z.f), len(z.g)) - 1)
    title = f"zoo: {z.name} | {z.description}"

    t0 = time.perf_counter()
    m = model.build(z.f, z.g, moments)
    p = portrait.certified_compute(
        m, view=view,
        trace_stable_branches=not args.no_stable)
    elapsed = time.perf_counter() - t0

    stem = f"{args.prefix}_{z.name}"
    plane_path = disk_path = None
    render_view = _fit_view_to_box(view, p.box)
    _maybe_report_view_clamp(view, render_view)
    if args.view_kind in ("plane", "both"):
        plane_path = str(out_dir / f"{stem}_plane.svg")
        _render_plane_path(p, plane_path, render_view, args, title)
    if args.view_kind in ("disk", "both"):
        disk_path = str(out_dir / f"{stem}_disk.svg")
        svg = render.disk_view(
            p, width=args.disk_size, height=args.disk_size,
            n_levels=max(1, args.levels // 2), title=title)
        render.save(svg, disk_path)

    summary = _case_summary(z.seed or 0, list(z.f), list(z.g), p, elapsed,
                            plane_path, disk_path, [], render_view)
    summary["zoo"] = {
        "name": z.name,
        "description": z.description,
        "discovery_seed": z.seed,
    }
    one_path = out_dir / f"{stem}_summary.json"
    one_path.write_text(json.dumps(summary, indent=2) + "\n")
    index_path = out_dir / f"{args.prefix}_zoo_index.json"
    index_path.write_text(json.dumps([summary], indent=2) + "\n")

    led = summary["ledger_summary"]
    print(
        f"[zoo:{z.name}] crit={summary['enumeration']['n_critical']} "
        f"branches={len(summary['branches'])} "
        f"Emax={led.get('worst_angle_energy', 0.0):.3e} "
        f"turn={led.get('worst_max_turn_deg', 0.0):.3f} "
        f"balanced={led.get('balanced')} "
        f"time={elapsed:.2f}s")
    if plane_path:
        print(f"  plane: {plane_path}")
    if disk_path:
        print(f"  disk:  {disk_path}")
    if args.no_stable:
        print("  stable separatrices skipped (--no-stable)")
    print(f"  json:  {one_path}")
    if args.auto_open:
        opened = _open_outputs([plane_path, disk_path], viewer=args.viewer)
        if opened:
            print(f"  opened {opened} SVG view{'s' if opened != 1 else ''}"
                  f" with viewer={args.viewer}")
    if args.pause:
        _pause_loop(p, out_dir, stem, title, args)
    print(f"index: {index_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spong-random-portrait",
        description="Generate random SPONG phase portraits for inspection.")
    p.add_argument("--seed", type=int, default=0,
                   help="master RNG seed")
    p.add_argument("--zoo", choices=zoo.names(), default=None,
                   help="render a named zoo portrait instead of random cases")
    p.add_argument("--count", type=int, default=1,
                   help="number of random portraits")
    p.add_argument("--pause", action="store_true",
                   help="open each generated portrait and wait between cases")
    p.add_argument("--open", dest="auto_open", action="store_true",
                   default=None,
                   help="open generated SVGs in the default browser/viewer")
    p.add_argument("--no-open", dest="auto_open", action="store_false",
                   help="do not open generated SVGs, even with --pause")
    p.add_argument("--viewer", choices=("auto", "browser", "inkscape", "open"),
                   default="auto",
                   help=("viewer for --open/--pause: auto uses a browser on "
                         "Windows; open uses the platform default"))
    p.add_argument("--same", action="store_true",
                   help="use g = f")
    p.add_argument("--f-degree", type=int, default=5,
                   help="degree of f")
    p.add_argument("--g-degree", type=int, default=5,
                   help="degree of g; ignored with --same")
    p.add_argument("--coeff-dist", choices=("normal", "uniform"),
                   default="normal",
                   help="coefficient distribution for f and g")
    p.add_argument("--coeff-scale", type=float, default=1.0,
                   help="coefficient scale")
    p.add_argument("--moment-dist", choices=("uniform01", "normal01"),
                   default="uniform01",
                   help="input distribution moments")
    p.add_argument("--view", nargs=4, type=float, default=None,
                   metavar=("A_LO", "A_HI", "B_LO", "B_HI"),
                   help="optional view box")
    p.add_argument("--view-kind", choices=("plane", "disk", "both"),
                   default="plane",
                   help="which SVG view(s) to write")
    p.add_argument("--output-dir", default="out/random_portraits",
                   help="directory for SVG and JSON outputs")
    p.add_argument("--prefix", default="portrait",
                   help="output filename prefix")
    p.add_argument("--levels", type=int, default=48,
                   help="number of plane contour levels")
    p.add_argument("--grid", type=int, default=1501,
                   help="b-grid size for plane contours")
    p.add_argument("--width", type=int, default=1200,
                   help="plane SVG width")
    p.add_argument("--height", type=int, default=900,
                   help="plane SVG height")
    p.add_argument("--disk-size", type=int, default=900,
                   help="disk SVG width and height")
    p.add_argument("--no-stable", action="store_true",
                   help="skip stable separatrices for quick visual scans")
    p.add_argument("--zoom-close", type=int, default=0,
                   help="write this many close-approach zoom plane SVGs")
    p.add_argument("--zoom-samples", type=int, default=1800,
                   help="samples per unstable branch for close-approach search")
    p.add_argument("--zoom-levels", type=int, default=24,
                   help="contour levels in close-approach zooms")
    p.add_argument("--zoom-grid", type=int, default=1001,
                   help="b-grid size for close-approach zoom contours")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.view = _view(args.view)
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.coeff_scale <= 0:
        parser.error("--coeff-scale must be positive")
    if args.same:
        args.g_degree = args.f_degree
    if args.auto_open is None:
        args.auto_open = args.pause
    if args.zoo is not None:
        return zoo_phase_portrait(args)
    return random_phase_portrait(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
