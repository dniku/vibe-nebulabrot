"""Command line interface: ``nebulabrot --output nebula.png``."""

from __future__ import annotations

import argparse
import pathlib
import sys

from nebulabrot import __version__, render
from nebulabrot.scene import DEFAULT_SCENE, Palette, Scene


def _palette(name: str) -> Palette:
    try:
        return Palette[name.upper()]
    except KeyError:
        names = ", ".join(member.name for member in Palette)
        raise argparse.ArgumentTypeError(
            f"unknown palette {name!r}, expected one of: {names}"
        ) from None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Nebulabrot: plot Mandelbrot orbits instead of escaping them.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("out/nebulabrot-3840x2160.png"),
        help="PNG file to write (default: %(default)s)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_SCENE.width)
    parser.add_argument("--height", type=int, default=DEFAULT_SCENE.height)
    parser.add_argument(
        "--center-real",
        type=float,
        default=DEFAULT_SCENE.center_real,
        help="real part of the viewport centre",
    )
    parser.add_argument(
        "--center-imag",
        type=float,
        default=DEFAULT_SCENE.center_imag,
        help="imaginary part of the viewport centre",
    )
    parser.add_argument(
        "--span-real",
        type=float,
        default=DEFAULT_SCENE.span_real,
        help="viewport width in complex units (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_SCENE.iterations,
        help="orbit length per sample; also the age range of the colour ramp",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SCENE.samples_per_pixel,
        help="jittered seeds per pixel (default: %(default)s)",
    )
    parser.add_argument(
        "--blur-taps",
        type=int,
        default=DEFAULT_SCENE.blur_taps,
        help="neighbour-averaging passes; 1 equals bilinear splatting (default: %(default)s)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=DEFAULT_SCENE.exposure,
        help="Reinhard exposure; higher brightens the faint outer clouds (default: %(default)s)",
    )
    parser.add_argument(
        "--palette",
        type=_palette,
        default=DEFAULT_SCENE.palette,
        metavar="{" + "|".join(member.name for member in Palette) + "}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SCENE.seed)
    parser.add_argument(
        "--jobs",
        type=int,
        help="worker processes; the default is chosen from the core count and free memory",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render the scene described by ``argv``; returns the process exit status."""
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    scene = Scene(
        width=args.width,
        height=args.height,
        center_real=args.center_real,
        center_imag=args.center_imag,
        span_real=args.span_real,
        iterations=args.iterations,
        samples_per_pixel=args.samples,
        blur_taps=args.blur_taps,
        exposure=args.exposure,
        palette=args.palette,
        seed=args.seed,
    )
    jobs = args.jobs
    if jobs is None:
        jobs = render.default_jobs(scene)
    elif jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    image = render.render(scene, jobs=jobs)
    render.write_png(image, scene, args.output)
    print(
        f"{args.output}: {scene.width}x{scene.height}, {scene.samples_per_pixel} samples/pixel, "
        f"{scene.iterations} iterations, {jobs} workers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
