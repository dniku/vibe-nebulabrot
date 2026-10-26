"""The artifact contract: same inputs, same bytes, whatever the machine did."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import PIL.Image

from nebulabrot import render
from nebulabrot.cli import _parse_args, scene_from_args
from nebulabrot.scene import DEFAULT_SCENE, Scene

#: A frame small enough to render in a test, taken from the shipped scene so that
#: the fields it does not care about cannot drift out of date.  Only the geometry,
#: the budget, and the tone window -- which belongs to the budget, and this frame
#: spends two orders of magnitude less light per pixel than the artifact does.
_PROBE: Scene = dataclasses.replace(
    DEFAULT_SCENE,
    width=96,
    height=54,
    center_real=-1.30,
    center_imag=-0.16,
    span_real=3.0,
    iterations=60,
    samples_per_pixel=4,
    blur_taps=1,
    tone_mid=12.0,
    chroma_neighbourhood=8,
    shadow_neighbourhood=2,
    anchor_real=-0.9,
)


def _scene(**overrides: object) -> Scene:
    return dataclasses.replace(_PROBE, **overrides)  # type: ignore[arg-type]


def _write(scene: Scene, jobs: int, path: pathlib.Path) -> str:
    render.write_png(render.render(scene, jobs=jobs), scene, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_running_with_no_flags_builds_the_shipped_scene() -> None:
    """The artifact is rendered through the command line, so its defaults have to
    be :data:`nebulabrot.scene.DEFAULT_SCENE` to the last bit.

    The scene travels into the PNG as a text chunk, so a default that the CLI
    re-encodes on the way through -- a palette angle turned into degrees and back
    -- changes the artifact's bytes while changing nothing visible.  Pixels
    identical, digest different, which is the worst kind of miss for a renderer
    whose whole claim is the digest.
    """
    assert scene_from_args(_parse_args([])) == DEFAULT_SCENE


def test_png_bytes_do_not_depend_on_worker_count(tmp_path: pathlib.Path) -> None:
    scene = _scene()
    digests = {_write(scene, jobs, tmp_path / f"{jobs}.png") for jobs in (1, 2, 5)}
    assert len(digests) == 1


def test_png_carries_no_varying_metadata_and_documents_the_scene(
    tmp_path: pathlib.Path,
) -> None:
    """A timestamped or path-dependent chunk would make the artifact unreproducible."""
    scene = _scene()
    path = tmp_path / "render.png"
    _write(scene, jobs=1, path=path)
    chunk_types = set()
    with path.open("rb") as handle:
        handle.read(8)  # PNG signature
        while chunk := handle.read(8):
            length = int.from_bytes(chunk[:4], "big")
            chunk_types.add(chunk[4:8])
            handle.seek(length + 4, 1)  # CRC
            handle.read(4)
    assert "tIME" not in chunk_types
    [metadata] = [PIL.Image.open(path).info["Description"]]
    assert json.loads(metadata) == {"renderer": "0.1.0", "scene": scene.metadata()}
    assert json.dumps(json.loads(metadata))  # round-trips, so it stays byte-stable


def test_rendered_frame_has_detail(tmp_path: pathlib.Path) -> None:
    """A frame worth looking at: real structure, mid-tone light, nothing flooded.

    This is the end-to-end smoke test of the pipeline: a mapping or accumulation
    regression that blanks or floods the frame fails here even when the
    individual stages still produce numbers.
    """
    image = render.render(
        _scene(width=256, height=144, iterations=200, samples_per_pixel=8), jobs=2
    )
    luma = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    assert luma.max() > 200, "the core should reach bright"
    assert (luma > 250).mean() < 0.05, "expected no blown-out highlights"
    assert np.percentile(luma, 20) < 80, "expected genuinely dark space"
    gradient = np.abs(np.diff(luma, axis=1)).mean()
    assert gradient > 1.0, "expected orbit structure, not a flat wash"


def test_frame_carries_many_hues_not_one() -> None:
    """The look this renderer exists for, stated as a check that can fail.

    Two separate wrong answers passed every check the artifact was originally
    given: a grey frame (luma statistics cannot see colour at all) and a frame of
    a single hue, which any ramp against orbit age produces, because nearly every
    pixel is lit by bounded orbits that run the whole iteration budget and so hold
    the same age distribution.  Hue coverage across the wheel is what separates
    those from the intended picture.
    """
    image = (
        render.render(
            _scene(width=256, height=144, iterations=200, samples_per_pixel=8), jobs=2
        ).astype(np.float64)
        / 255.0
    )
    maximum, minimum = image.max(axis=-1), image.min(axis=-1)
    lit = (maximum > 0.25) & (maximum - minimum > 0.05)
    assert lit.mean() > 0.2, "most of the frame should carry colour, not just the core"
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    hue = np.degrees(np.arctan2(green - 0.5 * (red + blue), (red - blue) / np.sqrt(3.0))) % 360.0
    histogram, _ = np.histogram(hue[lit], bins=12, range=(0.0, 360.0))
    share = histogram / histogram.sum()
    assert (share > 0.01).sum() >= 6, f"expected at least six hues, got {np.round(share, 3)}"
    assert share.max() < 0.6, f"no single hue should dominate: {np.round(share, 3)}"
