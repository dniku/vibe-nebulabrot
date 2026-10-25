"""Behaviour of the orbit accumulation, checked through the public renderer."""

from __future__ import annotations

import numpy as np
import pytest

from nebulabrot import render
from nebulabrot.colour import iteration_weights
from nebulabrot.scene import Palette, Scene


def _scene(**overrides: object) -> Scene:
    fields: dict[str, object] = {
        "width": 64,
        "height": 36,
        "center_real": -1.42,
        "center_imag": 0.0,
        "span_real": 3.2,
        "iterations": 60,
        "samples_per_pixel": 4,
        "blur_taps": 1,
        "exposure": 1.0,
        "palette": Palette.DEEP_FIELD,
        "seed": 0x5EED,
    }
    fields.update(overrides)
    return Scene(**fields)  # type: ignore[arg-type]


def test_first_orbit_point_lands_on_the_pixel_it_came_from() -> None:
    """``z_1 == c``, so one iteration must deposit exactly one splat per pixel.

    This pins the complex <-> pixel round trip: an off-by-one in either mapping
    would move light to a neighbouring pixel instead.
    """
    scene = _scene(iterations=1, samples_per_pixel=1, blur_taps=0)
    accumulated = render.accumulate(scene, jobs=1)
    [weight] = iteration_weights(scene.palette, 1)
    assert accumulated.shape == (scene.height, scene.width, 3)
    np.testing.assert_array_equal(accumulated, np.broadcast_to(weight, accumulated.shape))


def test_escaped_orbits_stop_contributing() -> None:
    """Well outside the set every orbit leaves the escape radius at ``z_2``.

    With ``c ~ -10`` the second point is ~90 units away, far beyond a viewport
    whose escape radius is ~12, so the render must be indistinguishable from the
    single-iteration render above.  Termination that fires too late would add
    light here, and termination that is too eager would remove the first splat.
    """
    scene = _scene(
        center_real=-10.0, span_real=3.0, iterations=40, samples_per_pixel=1, blur_taps=0
    )
    accumulated = render.accumulate(scene, jobs=1)
    [weight] = iteration_weights(scene.palette, 1)
    np.testing.assert_array_equal(accumulated, np.broadcast_to(weight, accumulated.shape))


@pytest.mark.parametrize("jobs", [1, 3, 5])
def test_partitioning_samples_does_not_change_the_image(jobs: int) -> None:
    """Accumulation is exact integer arithmetic, so it is associative.

    Workers and chunks split the samples differently, and the result must not
    notice: that is what makes a render reproducible across machines with a
    different core count.
    """
    assert np.array_equal(
        render.accumulate(_scene(), jobs=1), render.accumulate(_scene(), jobs=jobs)
    )


def test_blur_smooths_without_changing_total_light() -> None:
    """The tent filter redistributes light between neighbours and no further."""
    scene = _scene()
    unsmoothed = _scene(blur_taps=0)
    total = render.accumulate(unsmoothed, jobs=1).sum()
    smoothed = render.accumulate(scene, jobs=1).sum()
    # Only the outermost row/column of the tent loses light to the discard ring.
    assert smoothed == pytest.approx(total, rel=2 / scene.width)


def test_seed_changes_the_dust_but_not_the_total_light() -> None:
    """Jitter moves samples within a pixel; it must not add or remove light."""
    first = render.accumulate(_scene(seed=1), jobs=1)
    second = render.accumulate(_scene(seed=2), jobs=1)
    assert not np.array_equal(first, second)
    assert first.sum() == pytest.approx(second.sum(), rel=0.05)


def test_scene_rejects_impossible_settings() -> None:
    for field, value in [
        ("iterations", 0),
        ("samples_per_pixel", 0),
        ("span_real", 0.0),
        ("exposure", -1.0),
        ("blur_taps", -1),
    ]:
        with pytest.raises(ValueError):
            _scene(**{field: value})
