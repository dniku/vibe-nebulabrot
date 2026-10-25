"""Colour ramp, smoothing and tone mapping.

Orbit points are accumulated in linear light.  The per-iteration colour is
quantised to an exact integer multiple of ``1 / COLOUR_LEVELS`` so that every
partial sum the renderer performs is an exact integer: the accumulated buffer
is therefore independent of the order samples are processed in, which is what
lets a render be reproduced with any number of worker processes (see
:mod:`nebulabrot.kernel`).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nebulabrot.scene import Palette

#: Colour levels of the quantised weights.  16 bits keeps the accumulated
#: integer sum exact (float64 is exact for integers below 2**53) even if every
#: orbit point of a 4K render landed on a single pixel.
COLOUR_LEVELS = 1 << 16

#: Linear-light crossover of the sRGB transfer function (IEC 61966-2-1).
_SRGB_THRESHOLD = 0.0031308


def iteration_weights(scene_palette: Palette, iterations: int) -> npt.NDArray[np.float64]:
    """Quantised colour of orbit point ``i``, shape ``(iterations, 3)``.

    Orbit age is normalised over ``[0, 1]``, so the ramp is a property of the
    palette rather than of the iteration budget.  Rounding to multiples of
    ``1 / COLOUR_LEVELS`` is what makes every partial sum exact.
    """
    age = np.linspace(0.0, 1.0, iterations, dtype=np.float64)
    control_points = np.array(scene_palette.value, dtype=np.float64)
    ramp = np.stack(
        [np.interp(age, control_points[:, 0], column) for column in control_points[:, 1:].T]
    ).T
    return np.rint(ramp * COLOUR_LEVELS)


def smooth(accumulated: npt.NDArray[np.float64], taps: int, ring: int) -> npt.NDArray[np.float64]:
    """Average neighbours ``taps`` times, then crop the discard ring.

    A splat spread over the two pixels around its sub-pixel position is a
    2-tap tent per axis.  Accumulation is linear, so applying the tent once to
    the summed buffer is equivalent to applying it to every splat, at a small
    fraction of the cost.  ``taps == 1`` is exactly bilinear splatting.
    """
    blurred = accumulated
    for _ in range(taps):
        along_x = blurred * 0.5
        along_x[:, :-1, :] += blurred[:, 1:, :] * 0.5
        along_xy = along_x * 0.5
        along_xy[:-1, :, :] += along_x[1:, :, :] * 0.5
        blurred = along_xy
    return blurred[ring:-ring, ring:-ring, :]


def to_srgb(accumulated: npt.NDArray[np.float64], exposure: float) -> npt.NDArray[np.uint8]:
    """Tone-map linear accumulated light to 8-bit sRGB, in place.

    Reinhard compression (``x / (1 + x)``) keeps the dense core of the nebula
    from clipping while leaving the faint outer clouds visible, and saturates
    towards white exactly like a long exposure does.
    """
    linear = accumulated * (exposure / COLOUR_LEVELS)
    np.add(linear, 1.0, out=linear)
    np.reciprocal(linear, out=linear)
    np.subtract(1.0, linear, out=linear)
    encoded = np.where(
        linear <= _SRGB_THRESHOLD,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)
