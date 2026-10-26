"""Command line interface: ``nebulabrot --output nebula.png``."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

from nebulabrot import __version__, render
from nebulabrot.scene import DEFAULT_SCENE, Scene


def _degrees_as_radians(text: str) -> float:
    """Read a palette angle given in degrees as the radians the scene holds.

    ``argparse`` only ever applies ``type`` to text off the command line, never to
    a default, so leaving the defaults as scene values keeps them exact: run with
    no flags and the scene is reproduced bit for bit, which matters because the
    scene goes into the PNG as written metadata and the artifact is judged on its
    bytes.
    """
    return math.radians(float(text))


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
        help="orbit length per sample (default: %(default)s)",
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
        "--tone-mid",
        type=float,
        default=DEFAULT_SCENE.tone_mid,
        help="orbit points in a pixel that put it at half light (default: %(default)s)",
    )
    parser.add_argument(
        "--tone-decades",
        type=float,
        default=DEFAULT_SCENE.tone_decades,
        help="powers of ten of density, either side of --tone-mid, that set the "
        "slope of the tone curve (default: %(default)s)",
    )
    parser.add_argument(
        "--tone-floor",
        type=float,
        default=DEFAULT_SCENE.tone_floor,
        help="lightness of no light at all, which the tone curve approaches but "
        "never reaches (default: %(default)s, 0 sends empty space to black)",
    )
    parser.add_argument(
        "--coherence-shadow",
        type=float,
        default=DEFAULT_SCENE.coherence_shadow,
        help="how much a pixel that has lost its polarization is dimmed; this is "
        "what turns the interior of the set into a silhouette (default: %(default)s)",
    )
    parser.add_argument(
        "--chroma-gain",
        type=float,
        default=DEFAULT_SCENE.chroma_gain,
        help="fraction of a fully polarized pixel's lightness spent on colour "
        "(default: %(default)s, 0 gives a grey picture)",
    )
    parser.add_argument(
        "--shadow-neighbourhood",
        type=int,
        default=DEFAULT_SCENE.shadow_neighbourhood,
        help="radius in pixels the coherence shadow is measured over, which is "
        "much finer than --chroma-neighbourhood (default: %(default)s)",
    )
    parser.add_argument(
        "--chroma-gate",
        type=float,
        default=DEFAULT_SCENE.chroma_gate,
        help="exponent of the polarization in the chroma; below 1 keeps colour in "
        "the softly polarized middle of the frame too (default: %(default)s)",
    )
    parser.add_argument(
        "--chroma-neighbourhood",
        type=int,
        default=DEFAULT_SCENE.chroma_neighbourhood,
        help="radius in pixels of the neighbourhood a pixel's colour is judged "
        "against (default: %(default)s)",
    )
    parser.add_argument(
        "--anchor-real",
        type=float,
        default=DEFAULT_SCENE.anchor_real,
        help="real part of the pole the hue wheel turns about (default: %(default)s)",
    )
    parser.add_argument(
        "--anchor-imag",
        type=float,
        default=DEFAULT_SCENE.anchor_imag,
        help="imaginary part of the pole the hue wheel turns about",
    )
    parser.add_argument(
        "--harmonic",
        type=int,
        choices=(1, 2),
        default=DEFAULT_SCENE.harmonic,
        help="fold the circle of seed directions this many times before averaging "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--hue-offset",
        type=_degrees_as_radians,
        default=DEFAULT_SCENE.hue_offset,
        help="rotate the whole palette by this many degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--hue-dwell",
        type=float,
        default=DEFAULT_SCENE.hue_dwell,
        help="spend more of the frame on the hues around blue-violet and hurry "
        "through the warm side instead; -1 to 1 (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SCENE.seed)
    parser.add_argument(
        "--jobs",
        type=int,
        help="worker processes; the default is chosen from the core count and free memory",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args(argv)


def scene_from_args(args: argparse.Namespace) -> Scene:
    """The scene a set of parsed arguments describes.

    Names differ where the command line is worded for people: ``--samples`` means
    ``samples_per_pixel``, and ``--hue-offset`` is read in degrees.
    """
    return Scene(
        width=args.width,
        height=args.height,
        center_real=args.center_real,
        center_imag=args.center_imag,
        span_real=args.span_real,
        iterations=args.iterations,
        samples_per_pixel=args.samples,
        blur_taps=args.blur_taps,
        tone_mid=args.tone_mid,
        tone_decades=args.tone_decades,
        tone_floor=args.tone_floor,
        coherence_shadow=args.coherence_shadow,
        chroma_gain=args.chroma_gain,
        chroma_gate=args.chroma_gate,
        chroma_neighbourhood=args.chroma_neighbourhood,
        shadow_neighbourhood=args.shadow_neighbourhood,
        anchor_real=args.anchor_real,
        anchor_imag=args.anchor_imag,
        harmonic=args.harmonic,
        hue_offset=args.hue_offset,
        hue_dwell=args.hue_dwell,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """Render the scene described by ``argv``; returns the process exit status."""
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    scene = scene_from_args(args)
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
