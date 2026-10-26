"""Accumulation layout, smoothing, and turning moments into colour.

An accumulation buffer holds three moments per pixel: the orbit count, and the
real and imaginary parts of the accumulated flux vector described by
:class:`nebulabrot.scene.Scene`.  Two of the three are a *direction*, so the
buffer is not a colour and cannot simply be tone mapped: it has to be split into
the parts that make a picture.

    lightness  = characteristic curve of the orbit count, dimmed by a shadow of
                 how much of that light kept a common direction
    hue        = direction the light reaching the pixel came from, turned by the
                 palette
    chroma     = how much of that light kept a common direction

Both colour quantities are ratios, so they survive being compared over a
neighbourhood rather than per pixel: :func:`neighbourhood_average` blurs the
moments first, which is what makes the hue field vary smoothly across the frame
instead of pixel by pixel.  That separation -- crisp structure in lightness,
smooth structure in hue -- is the thing a single per-pixel density cannot do, and
the reason a Nebulabrot rendered as one channel and ramped through a palette
always comes out as one colour that only varies in brightness.

Lightness is driven by the orbit count and *not* by the flux magnitude, as the
reference colouring has it: the magnitude is the count times the polarization, so
using it for lightness blurs the density into the hue's neighbourhood and paints
the filaments out.  The polarization enters as a shadow instead -- a multiplication
in density, which is an addition in log density -- so the crisp structure the
count carries stays crisp while the interior of the set, where light arrives from
every direction at once, drops out as a silhouette.

Chroma is proportional to lightness, so a shadow keeps the hue of the light that
reaches it and only fades, rather than going grey.

Each channel of the buffer is an integer sum, which is what makes the render
bit-reproducible whatever ``--jobs`` does.  float64 represents integers below
2**53 exactly; :data:`nebulabrot.scene.FLUX_QUANTUM` bounds the worst case --
every orbit point of a 4K render landing in one pixel -- at 1.1e15.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from nebulabrot.scene import FLUX_QUANTUM, MOMENTS, Scene

#: Linear-light crossover of the sRGB transfer function (IEC 61966-2-1).
_SRGB_THRESHOLD = 0.0031308

# Oklab as published (https://bottosson.github.io/posts/oklab/), with the inverse
# matrices derived from these two rather than transcribed a second time.
_LINEAR_SRGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040792477],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_OKLAB_TO_LMS_ROOT = np.linalg.inv(_LMS_TO_OKLAB)
_LMS_TO_LINEAR_SRGB = np.linalg.inv(_LINEAR_SRGB_TO_LMS)

#: Rows of the image graded at a time.  The gamut search below needs a handful of
#: full-size temporaries per band; at 4K this keeps the peak to a few hundred MB.
_BAND_ROWS = 256


def smooth(accumulated: npt.NDArray[np.float64], taps: int, ring: int) -> npt.NDArray[np.float64]:
    """Average neighbours ``taps`` times, then crop the discard ring.

    A splat spread over the two pixels around its sub-pixel position is a
    2-tap tent per axis.  Accumulation is linear, so applying the tent once to
    the summed buffer is equivalent to applying it to every splat, at a small
    fraction of the cost.  ``taps == 1`` is exactly bilinear splatting.

    Every moment is blurred with the same kernel, which is what keeps the flux
    comparable with the count it is divided by.
    """
    blurred = accumulated
    for _ in range(taps):
        along_x = blurred * 0.5
        along_x[:, :-1, :] += blurred[:, 1:, :] * 0.5
        along_xy = along_x * 0.5
        along_xy[:-1, :, :] += along_x[1:, :, :] * 0.5
        blurred = along_xy
    return blurred[ring:-ring, ring:-ring, :]


def _moving_average(
    field: npt.NDArray[np.float64], radius: int, axis: int
) -> npt.NDArray[np.float64]:
    """Mean over a ``2 * radius + 1`` window, edge-replicated, in O(n)."""
    window = 2 * radius + 1
    border = [(0, 0)] * axis + [(radius, radius)] + [(0, 0)] * (field.ndim - axis - 1)
    padded = np.moveaxis(np.pad(field, border, mode="edge"), axis, 0)
    prefix = np.concatenate([np.zeros_like(padded[:1]), padded.cumsum(axis=0, dtype=np.float64)])
    summed = prefix[window : window + field.shape[axis]] - prefix[: field.shape[axis]]
    return np.moveaxis(summed, 0, axis) / window


def neighbourhood_average(
    field: npt.NDArray[np.float64], radius: int, passes: int = 3
) -> npt.NDArray[np.float64]:
    """Average each channel over a wide neighbourhood, three box passes.

    Three passes so the kernel is close enough to a Gaussian to leave no boxy
    ringing.  Edge replication keeps the frame border from being pulled towards
    zero, which a zero-padded filter would do as a dark or desaturated edge.
    """
    averaged = field
    for _ in range(passes):
        averaged = _moving_average(_moving_average(averaged, radius, 0), radius, 1)
    return averaged


def tone(
    signal: npt.NDArray[np.float64], *, mid: float, decades: float, floor: float
) -> npt.NDArray[np.float64]:
    """Map ``mid`` orbit points to half light, ``floor`` to no light and 1.0 to any light.

    A logistic on the logarithm of the signal -- the characteristic curve of
    photographic film.  Density has to be read logarithmically, since it spans
    orders of magnitude inside one frame and a linear ramp spends all of its
    steps on the brightest few pixels; but a log window that is then clipped to
    ``[0, 1]`` leaves two artefacts the eye reads at once.  Everything below the
    window clips to the same black, which turns the nebula's falloff into a flat
    mass -- 30% of the first 4K frame -- and everything above it has no way to
    reach white, which caps the same frame at two thirds light and gives it no
    hot cores at all.  The logistic has both ends as asymptotes instead: a
    density gradient stays a gradient however faint it gets, and
    ``decades``/``floor`` set its slope and its black without either clip.
    """
    windowed = np.log10(np.maximum(signal, 1e-12) / mid) / (2.0 * decades)
    return floor + (1.0 - floor) / (1.0 + 10.0 ** (-math.pi * windowed))


def _averaged(
    moments: npt.NDArray[np.float64], neighbourhood: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Count and flux vector of the light reaching each pixel's neighbourhood.

    The moments are averaged before any ratio is taken, so a pixel is measured
    against the light reaching its neighbourhood.  Division is the only way to
    get a scale-free measure out of a sum, and comparing a blurred numerator
    against an unblurred denominator would just move the noise somewhere else.
    ``neighbourhood == 0`` compares a pixel against nothing at all.
    """
    if not neighbourhood:
        return moments[..., 0], moments[..., 1:]
    return (
        neighbourhood_average(moments[..., 0], neighbourhood),
        neighbourhood_average(moments[..., 1:], neighbourhood),
    )


def _polarization(
    counts: npt.NDArray[np.float64], vectors: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Length of the mean flux vector: 1 for one direction, 0 for every direction.

    Back off the quantization first: the buffer holds whole numbers so that its
    sums are exact, and this is where they become vectors of length at most 1
    again.
    """
    magnitude = np.hypot(vectors[..., 0], vectors[..., 1]) * FLUX_QUANTUM
    return np.divide(magnitude, counts, out=np.zeros_like(magnitude), where=counts > 0.0)


def flux(
    moments: npt.NDArray[np.float64], *, neighbourhood: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Direction and polarization of the accumulated flux, per pixel.

    Polarization is also a confidence, which is why no separate noise gate is
    needed: where the vector sum is unstable because the contributions cancel, it
    is small by that same cancellation, so an unreliable hue never shows up as
    coloured grain.
    """
    counts, vectors = _averaged(moments, neighbourhood)
    return np.arctan2(vectors[..., 1], vectors[..., 0]), _polarization(counts, vectors)


def coherence(moments: npt.NDArray[np.float64], *, neighbourhood: int) -> npt.NDArray[np.float64]:
    """Polarization of the light reaching each pixel, on its own without the hue.

    The shadow that dims a pixel and the hue it is painted with ask the same
    question at two different scales, so they get their own measurement; see
    :attr:`nebulabrot.scene.Scene.shadow_neighbourhood`.
    """
    counts, vectors = _averaged(moments, neighbourhood)
    return _polarization(counts, vectors)


#: Oklab hue, in radians, that the palette lingers on: blue-violet, the hue the
#: eye is least able to place on the sphere and so the one a wide palette can
#: afford to spend its room on.  See :func:`palette_hue`.
DWELL_PHASE = math.radians(265.0)


def palette_hue(
    hue: npt.NDArray[np.float64], *, offset: float, dwell: float
) -> npt.NDArray[np.float64]:
    """Turn a measured flux direction into the hue the picture is painted with.

    The direction the light came from is a fair measurement and a poor palette:
    taken as a hue it spaces the spectrum evenly, and an even rainbow of broad,
    equally saturated fields is what makes a Nebulabrot look like a colour wheel
    rather than a nebula.

    ``offset`` rotates the palette as a whole.  ``dwell`` makes it non-uniform on
    the circle, spending more of the frame near :data:`DWELL_PHASE` and hurrying
    through the opposite, warm side; it must stay within ``[-1, 1]`` for the map
    to remain monotone.  The warp is periodic, which is the only way to space a
    hue circle unevenly without leaving a seam where it wraps -- scaling it would
    cut the frame in two along the anti-meridian.
    """
    turned = hue + offset
    return turned - dwell * np.sin(turned - DWELL_PHASE)


def oklab_to_linear_srgb(lab: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Oklab to linear sRGB, deliberately left unclipped.

    The gamut search below reads the negative excursions to tell how far outside
    the display cube a colour sits; clipping here would hide what it looks for.
    """
    lms_root = lab @ _OKLAB_TO_LMS_ROOT.T
    lms = np.sign(lms_root) * np.abs(lms_root) ** 3
    return lms @ _LMS_TO_LINEAR_SRGB.T


def _to_lch(
    lightness: npt.NDArray[np.float64],
    hue: npt.NDArray[np.float64],
    chroma: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1)


def gamut_map(
    lightness: npt.NDArray[np.float64],
    hue: npt.NDArray[np.float64],
    chroma: npt.NDArray[np.float64],
    iterations: int = 11,
) -> npt.NDArray[np.float64]:
    """Shrink chroma until the colour is displayable, holding hue and lightness.

    A binary search along the ray from the neutral axis, where leaving the display
    cube is monotone in chroma.  Clipping the channels instead would paint a hard
    edge wherever a gradient crossed the boundary of the gamut: a line across the
    middle of a nebula, in whatever hue happened to be outside.
    """
    low = np.zeros(lightness.shape, dtype=np.float64)
    high = np.ones(lightness.shape, dtype=np.float64)
    for _ in range(iterations):
        scale = (low + high) * 0.5
        rgb = oklab_to_linear_srgb(_to_lch(lightness, hue, chroma * scale))
        in_gamut = ((rgb >= 0.0) & (rgb <= 1.0)).all(axis=-1)
        low = np.where(in_gamut, scale, low)
        high = np.where(in_gamut, high, scale)
    return np.clip(oklab_to_linear_srgb(_to_lch(lightness, hue, chroma * low)), 0.0, 1.0)


def encode_srgb(linear: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Apply the sRGB transfer function and quantize to 8 bits."""
    encoded = np.where(
        linear <= _SRGB_THRESHOLD,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, 0.0, None), 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_srgb(moments: npt.NDArray[np.float64], scene: Scene) -> npt.NDArray[np.uint8]:
    """Tone map an accumulation buffer, ring already cropped, into 8-bit sRGB."""
    hue, polarization = flux(moments, neighbourhood=scene.chroma_neighbourhood)
    shadow = coherence(moments, neighbourhood=scene.shadow_neighbourhood)
    lightness = tone(
        moments[..., 0] * shadow**scene.coherence_shadow,
        mid=scene.tone_mid,
        decades=scene.tone_decades,
        floor=scene.tone_floor,
    )
    chroma = scene.chroma_gain * lightness * polarization**scene.chroma_gate
    hue = palette_hue(hue, offset=scene.hue_offset, dwell=scene.hue_dwell)
    encoded = np.empty((*lightness.shape, 3), dtype=np.uint8)
    for start in range(0, lightness.shape[0], _BAND_ROWS):
        stop = min(start + _BAND_ROWS, lightness.shape[0])
        encoded[start:stop] = encode_srgb(
            gamut_map(lightness[start:stop], hue[start:stop], chroma[start:stop])
        )
    return encoded


def blank_buffer(scene: Scene) -> npt.NDArray[np.float64]:
    """One accumulation buffer: three moments over the image plus its discard ring."""
    return np.zeros(
        (scene.height + 2 * scene.ring_width, scene.width + 2 * scene.ring_width, MOMENTS),
        dtype=np.float64,
    )
