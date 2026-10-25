"""Palette data integrity and the tone mapping curve."""

from __future__ import annotations

import numpy as np
import pytest

from nebulabrot.colour import COLOUR_LEVELS, iteration_weights, to_srgb
from nebulabrot.scene import Palette


@pytest.mark.parametrize("palette", list(Palette))
def test_palette_control_points_are_well_formed(palette: Palette) -> None:
    ages, *colours = np.array(palette.value, dtype=np.float64).T
    assert ages[0] == 0.0
    assert ages[-1] == 1.0
    assert np.all(np.diff(ages) > 0.0), "control points must ascend"
    for column in colours:
        assert np.all((column >= 0.0) & (column <= 1.0)), "linear-light values must stay in [0, 1]"


@pytest.mark.parametrize("palette", list(Palette))
def test_iteration_weights_stay_quantised_and_in_range(palette: Palette) -> None:
    weights = iteration_weights(palette, 512)
    assert weights.shape == (512, 3)
    # Exact integers are what make accumulation order-independent.
    np.testing.assert_array_equal(weights, np.rint(weights))
    assert weights.min() > 0.0
    assert weights.max() <= COLOUR_LEVELS


def test_tone_mapping_is_monotonic_and_end_to_end() -> None:
    """Reinhard never inverts, keeps black black, and asymptotes to white."""
    exposure = 1.0
    levels = np.linspace(0.0, 50.0 * _knee(exposure), 4096)
    grey = np.broadcast_to(levels.reshape(-1, 1, 1), (levels.size, 1, 3))
    image = to_srgb(grey, exposure=exposure, density_gamma=1.0)
    assert image.dtype == np.uint8
    assert np.all(image[:, 0, :] == image[:, 0, 0][:, None]), "channels must not diverge"
    assert np.all(np.diff(image[:, 0, 0]) >= 0), "more light must never darken a pixel"
    assert image[0, 0, 0] == 0
    assert image[-1, 0, 0] > 250


def test_exposure_positions_the_reinhard_knee() -> None:
    """``--exposure`` says how much accumulated light counts as the knee.

    One accumulated colour unit is ``1 / COLOUR_LEVELS`` of a fully bright
    splat, so the knee sits at ``COLOUR_LEVELS / exposure``.  Getting that scale
    wrong is the easy way to render an all-black or all-white frame.
    """
    for exposure in (0.00032, 0.01, 1.0, 40.0):
        at_knee = to_srgb(
            np.full((1, 1, 3), _knee(exposure)), exposure=exposure, density_gamma=1.0
        )[0, 0, 0]
        # Reinhard maps the knee to 0.5 linear light, which sRGB encodes as 186.
        assert 180 <= at_knee <= 192


def _knee(exposure: float) -> float:
    return COLOUR_LEVELS / exposure


def test_black_stays_black() -> None:
    assert to_srgb(np.zeros((1, 1, 3)), exposure=1234.5, density_gamma=0.7).tolist() == [
        [[0, 0, 0]]
    ]


def test_density_gamma_reveals_the_faint_end() -> None:
    """Orbit density spans decades; gamma < 1 is what makes the dust visible.

    Below the knee, warping with a gamma below 1 must lift dim light while
    leaving black at black, and the ordering of two light levels must not
    change: the curve may compress contrast but never invert it.
    """
    exposure = 0.0002
    dim, brighter = _knee(exposure) * np.array([1e-4, 1e-2])
    frame = np.zeros((2, 1, 3))
    frame[0, 0, :] = dim
    frame[1, 0, :] = brighter
    untouched = to_srgb(frame, exposure=exposure, density_gamma=1.0)[:, 0, 0]
    lifted = to_srgb(frame, exposure=exposure, density_gamma=0.5)[:, 0, 0]
    assert np.all(lifted > untouched), "gamma below 1 must lift light below the knee"
    assert lifted[0] < lifted[1], "the curve must stay monotonic"
    assert to_srgb(np.zeros((1, 1, 3)), exposure=exposure, density_gamma=0.5).max() == 0


def test_bright_light_washes_out_far_less_than_a_per_channel_roll_off() -> None:
    """Highlights must keep the hue of dim light of the same chromaticity.

    The sRGB transfer function is per channel, so encoded chromaticity drifts
    between two brightnesses even for a perfect roll-off.  Rolling Reinhard off
    per channel instead of on luminance drifts several times further, driving the
    dense core of a nebula towards white; that drift is the regression here, so
    the counterfactual is computed inline and compared against rather than a
    constant being asserted.
    """
    exposure = 0.0002
    tint = np.array([1.0, 0.5, 0.3])
    frame = np.stack([tint * _knee(exposure) * factor for factor in (0.02, 2.0)])[:, None, :]

    image = to_srgb(frame, exposure=exposure, density_gamma=1.0).astype(np.float64)
    drift = _chromaticity_drift(image)

    linear = frame / _knee(exposure)
    counterfactual = _encode_srgb(linear / (1.0 + linear))
    assert drift < _chromaticity_drift(counterfactual * 255.0) / 2.0


def _chromaticity_drift(image: np.ndarray) -> float:
    """Largest change in normalised RGB between the two pixels of ``image``."""
    chromaticity = image / image.max(axis=-1, keepdims=True)
    return float(np.abs(chromaticity[1, 0] - chromaticity[0, 0]).max())


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    """The transfer function, needed only to score the counterfactual roll-off."""
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(encoded, 0.0, 1.0)
