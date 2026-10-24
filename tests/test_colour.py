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
    linear = np.linspace(0.0, 4.0 * COLOUR_LEVELS, 4096)
    image = to_srgb(linear.reshape(-1, 1, 3), exposure=0.01)
    assert image.dtype == np.uint8
    assert np.all(np.diff(image[:, 0, 0]) >= 0), "more light must never darken a pixel"
    assert image[0, 0, 0] == 0
    assert image[-1, 0, 0] == 255


def test_black_stays_black() -> None:
    assert to_srgb(np.zeros((1, 1, 3)), exposure=1234.5).tolist() == [[[0, 0, 0]]]
