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

#: Colour levels of the quantised weights.  Accumulation stays exact because
#: every weight is an integer and float64 represents integers below 2**53
#: exactly.  The worst case is every orbit point of a render landing on one
#: pixel: the default 4K scene has 3840 * 2160 * 16 * 500 = 6.6e10 orbit points,
#: and 6.6e10 * 2**16 = 4.3e15 < 9.0e15, so even that keeps every sum exact.
COLOUR_LEVELS = 1 << 16

#: Linear-light crossover of the sRGB transfer function (IEC 61966-2-1).
_SRGB_THRESHOLD = 0.0031308

#: Rec. 709 luminance weights, used to roll off highlights without rotating hue.
LUMINANCE_COEFFICIENTS = np.array([0.2126, 0.7152, 0.0722])


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

    ``ring`` must be at least ``taps``: ``Scene.ring_width`` guarantees it, and
    each tap consumes one ring column that would otherwise have received light.
    """
    blurred = accumulated
    for _ in range(taps):
        along_x = blurred * 0.5
        along_x[:, :-1, :] += blurred[:, 1:, :] * 0.5
        along_xy = along_x * 0.5
        along_xy[:-1, :, :] += along_x[1:, :, :] * 0.5
        blurred = along_xy
    return blurred[ring:-ring, ring:-ring, :]


def to_srgb(
    accumulated: npt.NDArray[np.float64], exposure: float, density_gamma: float
) -> npt.NDArray[np.uint8]:
    """Tone-map accumulated linear light to 8-bit sRGB.

    ``density_gamma`` warps the density first, which is the part that makes a
    Nebulabrot viewable: orbit density spans five decades between the faint dust
    and the core, and a default frame has nothing in between.  Warping by
    ``x ** density_gamma`` spends the 8-bit range on the decades rather than on
    the core, at the cost of flattening contrast within any one of them.

    Reinhard is applied to luminance and then shared by all three channels, so a
    pixel keeps its hue as it brightens.  Applying it per channel instead drives
    each channel towards 1 independently, which measures as half the colour: on
    the default frame the brightest decile drops from C* 0.083 to 0.019.  The
    scale factor is ``1 / (1 + L)``, because ``L / (1 + L) / L`` collapses; at
    ``L == 0`` it is 1 and the channels are already zero.

    A saturated core can still outrun its own luminance, in which case both
    channels are scaled down together rather than clipped: 2.5% of the default
    frame is held at 255 that way, where clipping it would have desaturated those
    pixels towards white as well.
    """
    linear = np.power(accumulated * (exposure / COLOUR_LEVELS), density_gamma)
    roll_off = 1.0 / (1.0 + linear @ LUMINANCE_COEFFICIENTS)
    # A saturated highlight can outrun its own luminance.  Capping the shared
    # scale at ``1 / max channel`` brings it back into range proportionally, so
    # the pixel lands on 255 with its hue intact instead of clipping towards
    # white; the cap is inert below 1, where Reinhard is already in range.
    guard = 1.0 / np.maximum(linear.max(axis=-1), 1.0)
    linear *= np.expand_dims(np.minimum(roll_off, guard), axis=-1)
    encoded = np.where(
        linear <= _SRGB_THRESHOLD,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)
