"""Figures for the README, drawn from the scene the artifact is rendered from.

Not part of the package.  Regenerate everything the README embeds with::

    nix build .#image --out-link out      # the artifact, for the two showcase figures
    python docs/figures.py --artifact out --jobs 16

One accumulation of a small probe field is shared by all six panels of
``grades.png``, so the six differ only in the grade applied to them: about
twenty seconds, and a cache of the field is kept under the system temporary
directory.  Two of the panels draw curves the library no longer has -- the
clipped log window and the reference's coherent-flux lightness -- because those
are the wrong turns the figure exists to show.  They are implemented here, and
:func:`_encode` is asserted against ``colour.to_srgb`` on every run so a figure
cannot quietly stop resembling the renderer.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import tempfile

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from nebulabrot import colour, render
from nebulabrot.scene import DEFAULT_SCENE, Scene

#: Rec. 709 coefficients, for the one luminance statistic the labels quote.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

_CELL_WIDTH = 640
_LABEL_ROWS = 46


def _probe_scene() -> Scene:
    """The shipped framing at a size that accumulates in seconds."""
    return dataclasses.replace(DEFAULT_SCENE, width=1280, height=720, samples_per_pixel=16)


def _expose(moments: npt.NDArray[np.float64], scene: Scene) -> Scene:
    """``tone_mid`` for this probe's own sampling budget.

    ``tone_mid`` belongs to the budget rather than the geometry: a probe pixel
    spends a third of the artifact's samples over nine times the area of the
    parameter plane, so it holds a different amount of light, and the artifact's
    mid would burn the probe out. It is derived the same way the artifact's was
    measured -- the median density a little under half light -- and then held
    fixed for every panel, which is what makes the panels comparable.
    """
    shadow = colour.coherence(moments, neighbourhood=scene.shadow_neighbourhood)
    signal = moments[..., 0] * shadow**scene.coherence_shadow
    mid = float(np.median(signal)) / 10.0 ** (2.0 * scene.tone_decades * (0.42 - 0.5))
    return dataclasses.replace(scene, tone_mid=mid)


def _probe_field(scene: Scene, jobs: int, cache: pathlib.Path) -> npt.NDArray[np.float64]:
    try:
        return np.load(cache)
    except FileNotFoundError:
        accumulated = render.accumulate(scene, jobs=jobs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, accumulated)
        return accumulated


def _luma(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    encoded = image.astype(np.float64) / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    return linear @ _LUMA_WEIGHTS


def _detail(image: npt.NDArray[np.uint8]) -> float:
    """Mean edge energy of a panel, scaled to read as a few tens."""
    luminance = _luma(image)
    return 1e4 * float(
        np.abs(np.diff(luminance, axis=0)).mean() + np.abs(np.diff(luminance, axis=1)).mean()
    )


def _describe(image: npt.NDArray[np.uint8]) -> str:
    """What a panel says about itself, measured off its own bytes.

    ``detail`` is edge energy: it falls when something paints the filaments out,
    which is how the reference's coherent-flux lightness gives away that it is
    blurring the density it borrowed. ``flat`` is the largest group of pixels
    sharing one exact colour, which is what a clipped tone curve makes of every
    density under the clip -- 3% of this probe, and 30% of the 4K frame that
    first showed the defect. Both come off the encoded frame, because that is
    what the viewer sees.
    """
    pixels = image.reshape(-1, image.shape[2])
    _, counts = np.unique(pixels, axis=0, return_counts=True)
    flat = int(counts.max()) / pixels.shape[0]
    return f"detail {_detail(image):.0f}  flat mass {flat:.0%}"


def _encode(
    scene: Scene,
    lightness: npt.NDArray[np.float64],
    hue: npt.NDArray[np.float64],
    chroma: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint8]:
    """Gamut map and encode, the way :func:`nebulabrot.colour.to_srgb` does."""
    turned = colour.palette_hue(hue, offset=scene.hue_offset, dwell=scene.hue_dwell)
    encoded = np.empty((*lightness.shape, 3), dtype=np.uint8)
    for start in range(0, lightness.shape[0], 256):
        stop = min(start + 256, lightness.shape[0])
        encoded[start:stop] = colour.encode_srgb(
            colour.gamut_map(lightness[start:stop], turned[start:stop], chroma[start:stop])
        )
    return encoded


def _panels(
    moments: npt.NDArray[np.float64], scene: Scene
) -> list[tuple[str, npt.NDArray[np.uint8]]]:
    counts = moments[..., 0]
    hue, polarization = colour.flux(moments, neighbourhood=scene.chroma_neighbourhood)
    shadow = colour.coherence(moments, neighbourhood=scene.shadow_neighbourhood)
    shipped_signal = counts * shadow**scene.coherence_shadow

    def with_chroma(lightness: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return scene.chroma_gain * lightness * polarization**scene.chroma_gate

    # The curve the tone map had before the logistic: a log window clipped to the
    # displayable range, which pins everything below it to one value.
    clipped = np.maximum(
        np.clip(
            0.5
            + np.log10(np.maximum(shipped_signal, 1e-12) / scene.tone_mid)
            / (2.0 * scene.tone_decades),
            0.0,
            1.0,
        ),
        scene.tone_floor,
    )
    # The reference colouring: light with the magnitude of the flux sum, which is
    # the count times the polarization measured over the hue's wide neighbourhood.
    smoothed = colour.neighbourhood_average(counts, scene.chroma_neighbourhood)
    coherent = smoothed * polarization
    coherent_lightness = colour.tone(
        coherent,
        mid=float(np.median(coherent)) * 1.5,
        decades=scene.tone_decades,
        floor=scene.tone_floor,
    )
    shipped_lightness = colour.tone(
        shipped_signal,
        mid=scene.tone_mid,
        decades=scene.tone_decades,
        floor=scene.tone_floor,
    )
    return [
        (
            "Density only - one scalar, so one colour",
            _encode(scene, shipped_lightness, np.zeros_like(hue), np.zeros_like(shipped_lightness)),
        ),
        (
            "Coherence shadow off - the set glows",
            colour.to_srgb(moments, dataclasses.replace(scene, coherence_shadow=0.0)),
        ),
        (
            "Palette warp off - an even colour wheel",
            colour.to_srgb(moments, dataclasses.replace(scene, hue_dwell=0.0, hue_offset=0.0)),
        ),
        (
            "Reference lightness |sum| - detail smeared out",
            _encode(scene, coherent_lightness, hue, with_chroma(coherent_lightness)),
        ),
        (
            "Clipped log window - one colour under the clip",
            _encode(scene, clipped, hue, with_chroma(clipped)),
        ),
        ("Shipped", colour.to_srgb(moments, scene)),
    ]


def _tile(panels: list[tuple[str, npt.NDArray[np.uint8]]], columns: int) -> PIL.Image.Image:
    """Panels in a grid, each labelled with what it demonstrates."""
    rows = math.ceil(len(panels) / columns)
    sheet = PIL.Image.new(
        "RGB", (columns * _CELL_WIDTH, rows * (_CELL_WIDTH // 2 + _LABEL_ROWS)), (0, 0, 0)
    )
    title_font = PIL.ImageFont.load_default(size=21)
    stat_font = PIL.ImageFont.load_default(size=17)
    for index, (title, image) in enumerate(panels):
        column, row = index % columns, index // columns
        left = column * _CELL_WIDTH
        top = row * (_CELL_WIDTH // 2 + _LABEL_ROWS)
        draw = PIL.ImageDraw.Draw(sheet)
        draw.text((left + 6, top + 4), title, fill=(235, 235, 235), font=title_font)
        draw.text(
            (left + 6, top + 25),
            _describe(image),
            fill=(160, 160, 160),
            font=stat_font,
        )
        cell = PIL.Image.fromarray(image).resize((_CELL_WIDTH, _CELL_WIDTH // 2), PIL.Image.LANCZOS)
        sheet.paste(cell, (left, top + _LABEL_ROWS))
    return sheet


def _showcase(artifact: pathlib.Path, output: pathlib.Path) -> None:
    """The artifact itself, and a 1:1 crop of its filaments at the same scale."""
    image = PIL.Image.open(artifact)
    width, height = image.size
    image.resize((1920, height * 1920 // width), PIL.Image.LANCZOS).save(
        output / "artifact.png", optimize=True
    )
    side = min(1000, width, height)
    crop_box = (
        int(width * 0.30),
        int(height * 0.33),
        int(width * 0.30) + side,
        int(height * 0.33) + side,
    )
    image.crop(crop_box).save(output / "artifact-detail.png", optimize=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--jobs", type=int, default=8, help="workers for the probe accumulation")
    parser.add_argument(
        "--artifact",
        type=pathlib.Path,
        default=pathlib.Path("out/nebulabrot-3840x2160.png"),
        help="rendered 4K artifact to show off (render it with: nix build .#image)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("docs/images"),
        help="where to write",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path(tempfile.gettempdir()) / "nebulabrot-figures-probe.npy",
        help="cache of the probe accumulation",
    )
    parser.add_argument(
        "--skip-showcase",
        action="store_true",
        help="draw grades.png only, without the artifact figures",
    )
    args = parser.parse_args(argv)

    moments = _probe_field(_probe_scene(), jobs=args.jobs, cache=args.cache)
    scene = _expose(moments, _probe_scene())
    panels = _panels(moments, scene)

    # The panels that go through the library and the panels that go through the
    # local encoder have to agree, or the comparison in the figure is not one.
    library = colour.to_srgb(moments, scene)
    _, shipped = panels[-1]
    if not np.array_equal(library, shipped):
        raise SystemExit(
            "docs/figures.py no longer matches colour.to_srgb -- fix it before trusting it"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet = _tile(panels, columns=3)
    sheet.save(args.output_dir / "grades.png", optimize=True)
    print(f"{args.output_dir / 'grades.png'}: {sheet.size[0]}x{sheet.size[1]}")
    if not args.skip_showcase:
        _showcase(args.artifact, args.output_dir)
        print(f"{args.output_dir / 'artifact.png'} and artifact-detail.png written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
