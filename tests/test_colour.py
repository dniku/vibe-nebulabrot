"""Colour: density becomes light on a film curve, and the flux becomes hue."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from nebulabrot.colour import (
    _SRGB_THRESHOLD,
    DWELL_PHASE,
    encode_srgb,
    flux,
    gamut_map,
    neighbourhood_average,
    oklab_to_linear_srgb,
    palette_hue,
    smooth,
    to_srgb,
    tone,
)
from nebulabrot.scene import DEFAULT_SCENE, FLUX_QUANTUM, MOMENTS

#: The scale a flux vector is held at, in the same units as the orbit count: a
#: pixel lit from one direction accumulates one unit of vector per orbit point.
_FULLY_POLARIZED = 1.0 / FLUX_QUANTUM


def _pixel(counts: float, direction: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """A one-pixel accumulation: ``counts`` orbit points arriving from ``direction``.

    ``direction`` is the mean unit vector of the light, so ``(1, 0)`` is light
    that all came from one side of the pole and ``(0, 0)`` is light whose
    directions cancelled -- which is what a polarization of 0 and 1 look like.
    """
    moments = np.zeros((1, 1, MOMENTS))
    moments[0, 0] = (
        counts,
        counts * direction[0] * _FULLY_POLARIZED,
        counts * direction[1] * _FULLY_POLARIZED,
    )
    return moments


def _grade(moments: np.ndarray, **overrides: object) -> np.ndarray:
    scene = dataclasses.replace(DEFAULT_SCENE, **overrides)  # type: ignore[arg-type]
    return to_srgb(moments, scene)[0, 0]


def _channel_spread(rgb: np.ndarray) -> int:
    """A crude chroma: how far apart the channels ended up, in code values."""
    return int(rgb.max()) - int(rgb.min())


#: Oklab as published (https://bottosson.github.io/posts/oklab/), used to read
#: back what the grader wrote.
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040792477],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_LINEAR_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)


def to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Oklab of an 8-bit sRGB pixel."""
    encoded = rgb / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    return _LMS_TO_OKLAB @ np.cbrt(_LINEAR_SRGB_TO_LMS @ linear)


def lightness(rgb: np.ndarray) -> float:
    return float(to_oklab(rgb)[0])


def hue_angle(rgb: np.ndarray) -> float:
    lab = to_oklab(rgb)
    return float(np.arctan2(lab[2], lab[1]))


def oklab_linear(linear: np.ndarray) -> np.ndarray:
    """Oklab of a linear-light colour, before it is quantized to bytes."""
    return _LMS_TO_OKLAB @ np.cbrt(_LINEAR_SRGB_TO_LMS @ linear)


@pytest.mark.parametrize("counts", [0.0, 1.0, 1e4, 1e9])
@pytest.mark.parametrize("vector", [(0.0, 0.0), (1.0, 0.0), (-0.7, 0.7), (0.0, -1.0)])
def test_every_direction_and_density_encodes_inside_a_byte(
    counts: float, vector: tuple[float, float]
) -> None:
    """However hard the hue is asked for, no channel may be driven out of range."""
    graded = _grade(_pixel(counts, vector), chroma_gain=64.0)
    assert np.all(graded <= 255)


def test_a_pixel_with_no_light_sits_on_the_floor_and_carries_no_hue() -> None:
    graded = _grade(_pixel(0.0), tone_floor=0.07)
    assert lightness(graded) == pytest.approx(0.07, abs=0.01)
    assert _channel_spread(graded) == 0


def test_tone_reads_density_logarithmically() -> None:
    """A decade of density is worth as much light above the mid as it costs below.

    Orbit density runs over several orders of magnitude inside one frame, so the
    curve is written against its logarithm: what matters is the ratio to ``mid``,
    which makes the curve symmetric about mid on the linear scale it outputs.
    """
    window = tone(np.array([[1.0, 10.0, 100.0, 1000.0, 1e4]]), mid=100.0, decades=2.0, floor=0.0)[0]
    assert np.all(np.diff(window) > 0), "more density must never darken a pixel"
    assert window[2] == pytest.approx(0.5), "mid counts land at half light"
    assert window[1] + window[3] == pytest.approx(1.0), "a decade up pays a decade down"
    assert window[0] + window[4] == pytest.approx(1.0), "two decades up pays two decades down"


def test_tone_keeps_its_ends_as_asymptotes() -> None:
    """Faint light stays distinguishable from none, and bright light from white.

    A log window clipped to ``[0, 1]`` pins everything below the window onto the
    same value: 30% of the first 4K frame became one flat black mass, and the
    same clip from above capped it at two thirds light so that nothing in the
    frame could ever burn out to white.  Both ends have to be approached instead.
    """
    window = tone(
        np.array([[0.0, 1e-6, 1e-3, 100.0, 1e6, 1e12]]), mid=100.0, decades=1.0, floor=0.06
    )[0]
    assert window[0] == pytest.approx(0.06), "no light lands on the floor"
    assert np.all(np.diff(window) > 0), "every decade of density still moves the light"
    assert window[-1] < 1.0, "no amount of light reaches white"
    assert window[-1] > 0.999
    assert window[1] < 0.061, "the curve only just lifts off the floor down there"


def test_light_that_kept_its_direction_lights_a_pixel_more_than_light_that_did_not() -> None:
    """The coherence shadow: equal density, unequal light.

    Density alone lights the interior of the set as a glowing blob, because that
    is where the longest orbits pile up.  Light that arrives from every direction
    at once has nowhere to be said to come from, and dimming it by its own
    polarization is what turns that blob into a dark silhouette with dust lanes.
    """
    lit = _grade(_pixel(100.0, (1.0, 0.0)), tone_mid=100.0, shadow_neighbourhood=0)
    scattered = _grade(_pixel(100.0), tone_mid=100.0, shadow_neighbourhood=0)
    assert lightness(scattered) < 0.5 * lightness(lit)


def test_the_shadow_can_be_switched_off() -> None:
    lit = _grade(_pixel(100.0, (1.0, 0.0)), tone_mid=100.0, shadow_neighbourhood=0)
    scattered = _grade(_pixel(100.0), tone_mid=100.0, coherence_shadow=0.0, shadow_neighbourhood=0)
    assert lightness(scattered) == pytest.approx(lightness(lit), abs=0.01)


def test_the_palette_is_continuous_across_the_wrap() -> None:
    """A hue map with a seam would cut a line across the finished picture.

    Scaling a hue is the obvious way to space a palette unevenly and the obvious
    way to leave a discontinuity where it wraps, so the warp has to be periodic:
    a full turn in must be worth a full turn out, with no jump anywhere.
    """
    hue = np.linspace(-math.pi, math.pi, 2001)
    turned = palette_hue(hue, offset=0.4, dwell=0.3)
    assert turned[-1] - turned[0] == pytest.approx(2.0 * math.pi)
    assert np.all(np.diff(turned) > 0.0), "the map may not fold the circle back on itself"


def test_the_palette_packs_the_frame_onto_blue_violet() -> None:
    """The dwell spends more of the picture on one part of the spectrum.

    Taken as measured, the flux spaces the spectrum evenly, and an even rainbow of
    broad equally saturated fields is what makes a Nebulabrot read as a colour
    wheel.  The warp squeezes the hues around :data:`DWELL_PHASE` together, so
    more of the frame ends up sharing them, and stretches the opposite, warm side
    to pay for it.
    """
    span = math.radians(30.0)
    squeezed = palette_hue(
        np.array([DWELL_PHASE - span / 2, DWELL_PHASE + span / 2]), offset=0.0, dwell=0.6
    )
    stretched = palette_hue(
        np.array([DWELL_PHASE + math.pi - span / 2, DWELL_PHASE + math.pi + span / 2]),
        offset=0.0,
        dwell=0.6,
    )
    assert (stretched[1] - stretched[0]) > (squeezed[1] - squeezed[0]) * 3


def test_the_palette_offset_turns_the_whole_palette() -> None:
    hue = np.array([[0.0, 1.0, -2.0]])
    np.testing.assert_allclose(
        palette_hue(hue, offset=0.0, dwell=0.0) + 0.7,
        palette_hue(hue, offset=0.7, dwell=0.0),
    )


def test_light_reaching_a_pixel_from_one_side_of_the_pole_is_saturated() -> None:
    """Polarization is the whole of the colour: one direction, full hue."""
    graded = _grade(_pixel(100.0, (1.0, 0.0)), tone_mid=100.0, chroma_gain=0.6)
    assert _channel_spread(graded) > 60, f"expected a saturated pixel, got {graded}"


def test_light_whose_directions_cancel_is_grey() -> None:
    """Equal and opposite flux cancels, and where it cancels there is no hue.

    This is also the noise gate the design needed: an unstable hue arrives with
    the same cancellation that made it unstable, so it cannot show up as coloured
    grain.  Nothing has to be estimated or thresholded for that.
    """
    cancelling = _grade(_pixel(100.0), tone_mid=100.0, chroma_gain=0.6)
    assert cancelling == pytest.approx(np.full(3, cancelling[0]), abs=1.0)


def test_hue_follows_the_direction_the_light_came_from() -> None:
    """Turning the flux turns the hue, and half a turn swaps the colour.

    The flux is a direction, so this is what makes the picture two-dimensional
    rather than one number ramped through a palette.
    """
    opposite_hue = [
        _grade(_pixel(100.0, vector), tone_mid=100.0, chroma_gain=0.6)
        for vector in ((1.0, 0.0), (-1.0, 0.0))
    ]
    assert np.abs(opposite_hue[0].astype(int) - opposite_hue[1].astype(int)).max() > 40
    quarter_turn = [
        _grade(_pixel(100.0, vector), tone_mid=100.0, chroma_gain=0.6)
        for vector in ((1.0, 0.0), (0.0, 1.0))
    ]
    assert not np.array_equal(quarter_turn[0], quarter_turn[1])


def test_chroma_grows_with_lightness_so_shadows_keep_their_hue() -> None:
    """A dim pixel fades towards black holding its hue; it does not turn grey.

    The alternative -- capping chroma by how far the pixel is from mid light --
    was what made earlier frames read as monochrome in their darkest areas, which
    is where a Nebulabrot's colour is most visible.
    """
    dim = _grade(_pixel(35.0, (1.0, 0.0)), tone_mid=100.0, chroma_gain=0.6)
    bright = _grade(_pixel(400.0, (1.0, 0.0)), tone_mid=100.0, chroma_gain=0.6)
    assert lightness(dim) < lightness(bright)
    assert 0 < _channel_spread(dim) < _channel_spread(bright)
    for pair in (dim, bright):
        assert pair[2] > pair[1], f"both keep the blue side of the same hue: {pair}"


def test_gamut_map_trims_chroma_and_keeps_hue_and_lightness() -> None:
    """Out of gamut is met by pulling the colour back along its own ray.

    Clipping channels instead would put a hard edge wherever a gradient crossed
    the boundary of the displayable gamut: a line across the middle of a nebula,
    in whatever hue happened to be outside.
    """
    asked_lightness = np.full((1, 1), 0.55)
    hue = np.full((1, 1), 2.2)
    trimmed = gamut_map(asked_lightness, hue, np.full((1, 1), 0.9))[0, 0]
    assert np.all((trimmed >= 0.0) & (trimmed <= 1.0))
    assert np.abs(trimmed - trimmed.mean()).max() < 0.45, (
        "the chroma must actually have been trimmed"
    )
    passed_through = oklab_to_linear_srgb(np.array([[0.55, 0.9 * np.cos(2.2), 0.9 * np.sin(2.2)]]))[
        0
    ]
    assert np.abs(trimmed - passed_through).max() > 0.1, (
        "and it cannot just have been passed through"
    )

    # A request well inside the gamut is returned untouched: the search must not
    # cost anything from a colour that had room.
    comfortable = gamut_map(asked_lightness, hue, np.full((1, 1), 0.02))[0, 0]
    assert comfortable == pytest.approx(
        oklab_to_linear_srgb(np.array([[0.55, 0.02 * np.cos(2.2), 0.02 * np.sin(2.2)]]))[0],
        abs=1e-4,
    ), "the search stops one quantum short of full chroma"

    trimmed_lab = oklab_linear(trimmed)
    comfortable_lab = oklab_linear(comfortable)
    assert trimmed_lab[0] == pytest.approx(comfortable_lab[0], abs=0.002), (
        "trimming chroma must not move lightness"
    )
    assert np.arctan2(trimmed_lab[2], trimmed_lab[1]) == pytest.approx(
        np.arctan2(comfortable_lab[2], comfortable_lab[1]), abs=1e-6
    ), "trimming chroma must not turn the hue"
    assert np.hypot(trimmed_lab[1], trimmed_lab[2]) < 0.9, "the chroma was the only thing moved"


@pytest.mark.parametrize("linear", [_SRGB_THRESHOLD * 0.5, 0.2, 1.0])
def test_transfer_function_is_the_sRGB_one(linear: float) -> None:
    """The linear segment of sRGB is what keeps dark dust from banding away."""
    encoded = encode_srgb(np.array([[linear]]))[0, 0] / 255.0
    expected = (
        linear * 12.92 if linear <= _SRGB_THRESHOLD else 1.055 * linear ** (1.0 / 2.4) - 0.055
    )
    assert encoded == pytest.approx(expected, abs=1.0 / 510.0)


def test_smooth_blurs_neighbours_and_crops_the_discard_ring() -> None:
    size = 6
    ring = 1
    accumulated = np.zeros((size, size, MOMENTS))
    accumulated[2, 2] = (4.0, 8.0, 12.0)
    blurred = smooth(accumulated, taps=1, ring=ring)
    assert blurred.shape == (size - 2 * ring, size - 2 * ring, MOMENTS)
    for channel in range(MOMENTS):
        plane = blurred[..., channel]
        assert plane.sum() == pytest.approx(accumulated[..., channel].sum()), (
            "the tent only moves light around"
        )
        assert plane.max() < accumulated[..., channel].max(), "a hot value must spread"


def test_neighbourhood_average_holds_a_constant_field_and_its_borders() -> None:
    """Edge replication, because zero padding would eat the frame's own border."""
    constant = np.full((9, 11, 2), 3.5)
    averaged = neighbourhood_average(constant, radius=3)
    assert np.all(averaged == pytest.approx(3.5))
    total = np.zeros((9, 11, 2))
    total[4, 5] = 8.0
    spread = neighbourhood_average(total, radius=2)
    assert spread.max() < total.max(), "a hot value must spread"
    assert spread[4, 5, 0] > spread[0, 0, 0], "and thin out away from it"


def test_polarization_is_measured_over_a_neighbourhood() -> None:
    """A pixel is judged against the light reaching around it, not alone.

    Dividing a blurred flux by a blurred count is what makes the hue field vary
    smoothly across the frame; comparing a pixel against nothing turns sampling
    noise straight into coloured grain.
    """
    moments = np.zeros((9, 9, MOMENTS))
    moments[..., 0] = 10.0
    moments[..., 1] = 10.0 * _FULLY_POLARIZED
    sharp_hue, sharp_polarization = flux(moments, neighbourhood=0)
    _, smooth_polarization = flux(moments, neighbourhood=4)
    assert sharp_polarization[4, 4] == pytest.approx(1.0)
    assert smooth_polarization[4, 4] == pytest.approx(1.0)
    moments[4, 4, 1] = 0.0
    _, mixed = flux(moments, neighbourhood=4)
    assert mixed[4, 4] < 1.0 - 1e-6, "a cancelled pixel must be diluted by its neighbours"
    del sharp_hue


def test_oklab_inverse_matches_the_published_forward_transform() -> None:
    """The inverse matrices are derived from the published ones, not retyped.

    Oklab of linear red is published alongside the transform, so the inverse is
    checked against it rather than against itself.
    """
    published_red = np.array([[0.628_001_1, 0.224_898_3, 0.125_857_5]])
    assert oklab_to_linear_srgb(published_red)[0] == pytest.approx((1.0, 0.0, 0.0), abs=5e-4)
