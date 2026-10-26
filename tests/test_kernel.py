"""Behaviour of the orbit accumulation, checked through the public renderer."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from nebulabrot import render
from nebulabrot.scene import DEFAULT_SCENE, FLUX_QUANTUM, MOMENTS, Scene

#: A frame small enough to accumulate in a test, taken from the shipped scene so
#: that the grade fields it does not care about cannot drift out of date.
_PROBE: Scene = dataclasses.replace(
    DEFAULT_SCENE,
    width=64,
    height=36,
    center_real=-1.42,
    center_imag=0.0,
    span_real=3.2,
    iterations=60,
    samples_per_pixel=4,
    blur_taps=1,
)


def _scene(**overrides: object) -> Scene:
    return dataclasses.replace(_PROBE, **overrides)  # type: ignore[arg-type]


def _polarization(accumulated: np.ndarray) -> np.ndarray:
    """Length of the mean flux vector per pixel, in the pixels that hold light."""
    counts = accumulated[..., 0]
    magnitude = np.hypot(accumulated[..., 1], accumulated[..., 2]) * FLUX_QUANTUM
    return np.divide(magnitude, counts, out=np.zeros_like(counts), where=counts > 0.0)


def test_the_first_orbit_point_is_not_drawn() -> None:
    """``z_1 == c``, so drawing it would lay one point on the pixel it came from.

    One point per sample, over every pixel in the frame, is a flat field: as
    bright as the faintest dust the picture is made of, and its flux vector points
    wherever that pixel sits relative to the pole, which paints empty space in a
    wash of its own position angle.
    """
    accumulated = render.accumulate(_scene(iterations=1, samples_per_pixel=1, blur_taps=0), jobs=1)
    assert accumulated.shape == (36, 64, MOMENTS)
    np.testing.assert_array_equal(accumulated, np.zeros(accumulated.shape))


def test_bounded_orbits_are_not_drawn() -> None:
    """A viewport inside the set holds light for the whole budget and paints nothing.

    A bounded orbit never leaves the frame, so it spends every remaining
    iteration on its attractor.  Drawn with the escaping orbits it buries the
    nebula under the silhouette of the set, which is why the budget decides
    whether a sample counts at all.
    """
    inside_the_set = _scene(
        center_real=-0.5, center_imag=0.0, span_real=0.02, samples_per_pixel=1, blur_taps=0
    )
    np.testing.assert_array_equal(
        render.accumulate(inside_the_set, jobs=1), np.zeros((36, 64, MOMENTS))
    )


def test_orbits_that_leave_the_frame_stop_contributing() -> None:
    """Well outside the set an orbit is gone by ``z_2``, so nothing is drawn.

    With ``c ~ -10`` the second point is ~90 units away, far beyond a viewport
    whose escape radius is ~11, and the render cannot tell the difference between
    that and a single iteration.  Termination that fired too late would add light
    here; termination that was too eager would take away the second point of the
    orbits that linger.
    """
    far_from_the_set = _scene(
        center_real=-10.0, span_real=3.0, iterations=40, samples_per_pixel=1, blur_taps=0
    )
    np.testing.assert_array_equal(
        render.accumulate(far_from_the_set, jobs=1), np.zeros((36, 64, MOMENTS))
    )


def test_a_seed_on_the_pole_carries_no_vector_but_still_lights_its_pixel() -> None:
    """The pole has no direction, so a seed there contributes density and no hue.

    Silently letting it carry a vector of whatever length the division produced
    would put an arbitrary hue in the middle of the composition.
    """
    scene = _scene(anchor_real=-0.75, anchor_imag=0.25)
    real, imag = scene.flux_vector(np.array([-0.75]), np.array([0.25]))
    assert (real[0], imag[0]) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [((1.0, 0.0), (1.0, 0.0)), ((0.0, -1.0), (0.0, -1.0)), ((-1.0, 0.0), (-1.0, 0.0))],
)
def test_the_flux_vector_points_from_the_pole_to_the_seed(
    seed: tuple[float, float], expected: tuple[float, float]
) -> None:
    scene = _scene(anchor_real=0.0, anchor_imag=0.0)
    real, imag = scene.flux_vector(np.array([seed[0]]), np.array([seed[1]]))
    assert real[0] * FLUX_QUANTUM == pytest.approx(expected[0], abs=1e-3)
    assert imag[0] * FLUX_QUANTUM == pytest.approx(expected[1], abs=1e-3)


def test_the_vector_stays_unit_whatever_the_distance_to_the_pole() -> None:
    """Only direction is accumulated, so distance must not weigh a contribution.

    Weighting by distance instead would make the hue field track how far the
    frame's seeds sit from the pole, which is a brightness-shaped gradient over
    the whole picture.
    """
    scene = _scene(anchor_real=-1.0, anchor_imag=0.5)
    real, imag = scene.flux_vector(
        np.array([-0.4, -0.55, -1.42, -3.0, 0.3]), np.array([-2.25, -2.35, -2.45, -3.0, -1.0])
    )
    length = np.hypot(real, imag) * FLUX_QUANTUM
    lit = length > 0.0
    assert lit.all()
    np.testing.assert_allclose(length[lit], 1.0, atol=1e-3)


def test_the_second_harmonic_folds_the_circle_once() -> None:
    """Directions a half turn apart become the same direction, and opposite ones agree.

    That folding is what moves where the flux cancels, which is the difference
    between two colourings of the same render.
    """
    half_turn_apart = np.array([[1.0, -1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    folded = _vectors(_scene(harmonic=2, anchor_real=0.0, anchor_imag=0.0), half_turn_apart)
    straight = _vectors(_scene(harmonic=1, anchor_real=0.0, anchor_imag=0.0), half_turn_apart)
    np.testing.assert_allclose(straight[0], -straight[1], atol=1e-3)
    np.testing.assert_allclose(folded, [[0.0, -1.0], [0.0, -1.0]], atol=1e-3)


def _vectors(scene: Scene, seeds: np.ndarray) -> np.ndarray:
    real, imag = scene.flux_vector(seeds[:, 0], seeds[:, 1])
    return np.stack([real, imag], axis=-1) * FLUX_QUANTUM


def test_light_from_one_side_of_the_pole_is_fully_polarized() -> None:
    """A frame whose every seed lies one way from the pole must cancel nothing.

    The whole colour design rests on the flux being a real mixture, so this is
    the degenerate case where it is not, checked end to end rather than on the
    vector helper alone.
    """
    one_sided = _scene(anchor_real=5.0, anchor_imag=0.0)
    accumulated = render.accumulate(one_sided, jobs=2)
    lit = accumulated[..., 0] > 0
    assert lit.any()
    assert (accumulated[..., 1][lit] < 0).all(), "every seed lies to the left of the pole"
    # Not exactly 1: the seeds of one pixel straddle the pole's direction by a
    # little, and the deposited vectors are quantized.
    assert _polarization(accumulated)[lit].min() > 0.99, (
        f"a one-sided frame must barely cancel: {_polarization(accumulated)[lit].min()}"
    )


@pytest.mark.parametrize("jobs", [1, 3, 5])
def test_partitioning_samples_does_not_change_the_buffer(jobs: int) -> None:
    """Accumulation is exact integer arithmetic, so it is associative.

    Workers and chunks split the samples differently, and the result must not
    notice: that is what makes a render reproducible across machines with a
    different core count.  It holds of the flux as much as of the count, which is
    why the deposited vectors are quantized onto an integer grid.
    """
    assert np.array_equal(
        render.accumulate(_scene(), jobs=1), render.accumulate(_scene(), jobs=jobs)
    )


def test_blur_smooths_without_changing_total_light() -> None:
    """The tent filter redistributes light between neighbours and no further."""
    counts = render.accumulate(_scene(blur_taps=0), jobs=1)[..., 0]
    smoothed = render.accumulate(_scene(), jobs=1)[..., 0]
    # Only the outermost row/column of the tent loses light to the discard ring.
    assert smoothed.sum() == pytest.approx(counts.sum(), rel=2 / _scene().width)


def test_seed_changes_the_dust_but_not_the_amount_of_light() -> None:
    """Jitter moves samples within a pixel, so the total is only stable at scale.

    A jittered seed can escape one iteration sooner or later than its neighbour,
    and near the set boundary that difference is large, so the totals agree to a
    few per cent rather than exactly.  What must hold exactly is that the same
    seed always gives the same buffer, which the partitioning test covers.
    """
    first = render.accumulate(_scene(seed=1), jobs=1)
    second = render.accumulate(_scene(seed=2), jobs=1)
    assert not np.array_equal(first, second)
    assert first[..., 0].sum() == pytest.approx(second[..., 0].sum(), rel=0.05)


def test_scene_rejects_impossible_settings() -> None:
    for field, value in [
        ("iterations", 0),
        ("samples_per_pixel", 0),
        ("span_real", 0.0),
        ("tone_mid", 0.0),
        ("tone_decades", 0.0),
        ("tone_floor", -0.1),
        ("tone_floor", 1.0),
        ("chroma_gain", -1.0),
        ("chroma_gate", -0.1),
        ("chroma_gate", 1.1),
        ("coherence_shadow", -0.1),
        ("chroma_neighbourhood", -1),
        ("shadow_neighbourhood", -1),
        ("hue_dwell", 1.5),
        ("hue_dwell", -1.5),
        ("blur_taps", -1),
        ("harmonic", 0),
        ("harmonic", 3),
        ("center_real", float("nan")),
        ("anchor_imag", float("inf")),
    ]:
        with pytest.raises(ValueError):
            _scene(**{field: value})
