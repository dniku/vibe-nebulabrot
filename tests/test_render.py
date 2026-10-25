"""The artifact contract: same inputs, same bytes, whatever the machine did."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import PIL.Image

from nebulabrot import render
from nebulabrot.scene import DEFAULT_SCENE, Palette, Scene


def _scene(**overrides: object) -> Scene:
    fields: dict[str, object] = {
        "width": 96,
        "height": 54,
        "center_real": -1.30,
        "center_imag": -0.16,
        "span_real": 3.0,
        "iterations": 60,
        "samples_per_pixel": 4,
        "blur_taps": 1,
        "exposure": 0.0002,
        "density_gamma": 0.7,
        "palette": Palette.DEEP_FIELD,
        "seed": 0x5EED,
    }
    fields.update(overrides)
    return Scene(**fields)  # type: ignore[arg-type]


def _write(scene: Scene, jobs: int, path: pathlib.Path) -> str:
    render.write_png(render.render(scene, jobs=jobs), scene, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """A frame worth looking at: structure in the mid-tones, and real black space.

    This is the end-to-end smoke test of the pipeline: a mapping or accumulation
    regression that blanks or floods the frame fails here even when the
    individual stages still produce numbers.
    """
    image = render.render(
        _scene(width=256, height=144, iterations=200, samples_per_pixel=8), jobs=2
    )
    luma = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    assert 0.02 < (luma < 5).mean() < 0.9, "expected some genuinely black sky"
    assert (luma > 250).mean() < 0.05, "expected no blown-out highlights"
    gradient = np.abs(np.diff(luma, axis=1)).mean()
    assert gradient > 1.0, "expected orbit structure, not a flat wash"


def test_default_palette_cannot_render_grey() -> None:
    """A grey artifact passed every check it was originally tuned against.

    Luma statistics cannot see colour, and hue-per-age averages the ramp over
    whichever ages reach a pixel, so a ramp that spans complementary hues
    averages towards neutral.  A ramp inside one hue family cannot, which is why
    the default ramp does; the margin between the two cases is 0.07 against 0.21
    on this scene.
    """
    scene = dataclasses.replace(
        DEFAULT_SCENE, width=96, height=54, iterations=60, samples_per_pixel=4
    )
    image = render.render(scene, jobs=1).astype(np.float64) / 255.0
    visible = image.max(axis=-1) > 0.2
    assert visible.mean() > 0.05, "the scene has to put light on some pixels"
    assert (image.max(axis=-1) - image.min(axis=-1))[visible].mean() > 0.15
