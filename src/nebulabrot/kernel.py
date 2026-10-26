"""Orbit accumulation: the compute-bound inner loop of the renderer.

A Nebulabrot plots orbits instead of escaping them.  Every sample is a point
``c`` of the Mandelbrot parameter plane; its orbit ``z <- z^2 + c`` is iterated
and each ``z`` drops light into whichever pixel of the *same* viewport it
lands on.  Orbits that linger paint glowing clouds, and orbits that escape paint
the rays that cross the frame.

Only orbits that escape are drawn.  A bounded orbit never leaves the frame, so it
spends its entire remaining iteration budget piling onto its own attractor; drawn
alongside the escaping ones it floods the interior of the set with light and
buries the nebula.  Gated to the escaping orbits, the set reads as a dark
silhouette against the glow of the orbits that got away, which is the picture.
Finding out which orbits escape costs a pass over the arithmetic with nothing
plotted, and it retires the quarter of the frame that never comes back before the
splatting pass, so the two passes together cost about what one drawn pass did.

Each deposit carries three moments of the sample, described by
:meth:`nebulabrot.scene.Scene.flux_vector`: the orbit count that becomes
lightness, and the real and imaginary parts of the unit vector from the hue
pole to the seed, which become colour.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nebulabrot.scene import MOMENTS, Scene

#: Samples accumulated at a time per worker.  Small enough that the live
#: sample arrays stay in cache while the buffers are being compacted.
CHUNK_SAMPLES = 1 << 16

_UINT64 = np.uint64
_GOLDEN_RATIO = _UINT64(0x9E3779B97F4A7C15)
_SHIFT_A = _UINT64(30)
_SHIFT_B = _UINT64(27)
_SHIFT_C = _UINT64(31)
_MULTIPLIER_A = _UINT64(0xBF58476D1CE4E5B9)
_MULTIPLIER_B = _UINT64(0x94D049BB133111EB)
#: uint64 -> float in [0, 1): keep the top 53 bits, which a float64 holds exactly.
_TO_UNIT_SHIFT = _UINT64(11)
_UINT64_TO_UNIT = 2.0**-53


def _splitmix64(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Vectorised SplitMix64 finaliser (public domain, Sturm 2017)."""
    mixed = state + _GOLDEN_RATIO
    mixed = (mixed ^ (mixed >> _SHIFT_A)) * _MULTIPLIER_A
    mixed = (mixed ^ (mixed >> _SHIFT_B)) * _MULTIPLIER_B
    return mixed ^ (mixed >> _SHIFT_C)


def _jitter(
    sample_index: npt.NDArray[np.uint64], seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sub-pixel offsets in ``[-0.5, 0.5)`` for the given sample indices.

    The offsets are a pure function of ``(seed, sample index)`` rather than a
    pull from a random stream, so they do not depend on how samples are split
    between chunks or worker processes.  That keeps the render bit-identical
    whatever the value of ``--jobs`` is.
    """
    key = sample_index * _UINT64(2) + (_UINT64(seed) ^ _GOLDEN_RATIO)
    real = (_splitmix64(key) >> _TO_UNIT_SHIFT).astype(np.float64) * _UINT64_TO_UNIT
    imag = (_splitmix64(key + _UINT64(1)) >> _TO_UNIT_SHIFT).astype(np.float64) * _UINT64_TO_UNIT
    return real - 0.5, imag - 0.5


def _iterate(
    z_real: npt.NDArray[np.float64],
    z_imag: npt.NDArray[np.float64],
    c_real: npt.NDArray[np.float64],
    c_imag: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One step of ``z <- z^2 + c``.

    Both passes step through here, so the gate cannot disagree with what gets
    drawn: a sample the gate retires is a sample the drawing pass would have
    retired on the same iteration.
    """
    return (
        z_real * z_real - z_imag * z_imag + c_real,
        z_real * z_imag + z_real * z_imag + c_imag,
    )


def _escapes(
    scene: Scene, c_real: npt.NDArray[np.float64], c_imag: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Which of these seeds orbit out of the escape radius within the budget.

    ``z`` is compacted against the escape test exactly as it is while drawing, so
    the cost falls away as the samples that escape early drop out; a sample is
    retired the moment ``|z|`` exceeds the radius, and NaN or overflow retires it
    too, since ``<=`` is false for both.
    """
    escape_radius_squared = scene.escape_radius_squared
    live = np.arange(c_real.size, dtype=np.intp)
    escaped = np.zeros(c_real.shape, dtype=bool)
    z_real = np.zeros(c_real.shape, dtype=np.float64)
    z_imag = np.zeros(c_imag.shape, dtype=np.float64)
    for _ in range(scene.iterations):
        z_real, z_imag = _iterate(z_real, z_imag, c_real, c_imag)
        inside = (z_real * z_real + z_imag * z_imag) <= escape_radius_squared
        if not inside.all():
            escaped[live[~inside]] = True
            live = live[inside]
            z_real, z_imag = z_real[inside], z_imag[inside]
            c_real, c_imag = c_real[inside], c_imag[inside]
        if live.size == 0:
            break
    return escaped


def _deposit_orbits(
    scene: Scene,
    buffer: npt.NDArray[np.float64],
    vector_real: npt.NDArray[np.float64],
    vector_imag: npt.NDArray[np.float64],
    c_real: npt.NDArray[np.float64],
    c_imag: npt.NDArray[np.float64],
) -> None:
    """Drop the orbit points of the given samples into the accumulation buffer.

    ``buffer`` covers the image plus a discard ring of ``scene.ring_width``
    pixels, one row per pixel and the three moments down its columns.  Orbit
    points that leave the viewport are clamped into the ring, which is dropped by
    :func:`nebulabrot.colour.smooth`.

    Drawing starts at ``z_2``.  ``z_1`` is ``c``, so plotting it lays exactly one
    point on the very pixel the sample came from, everywhere in the frame at
    once: a flat field that carries no information about any orbit, that is as
    bright as the nebula's faintest dust, and whose flux vector points wherever
    that pixel sits relative to the pole -- which paints empty space in a smooth
    wash of its own position angle.
    """
    escape_radius_squared = scene.escape_radius_squared
    ring = scene.ring_width
    padded_width = scene.width + 2 * ring
    # Orbit points outside the viewport are parked in the buffer's outermost
    # index, which the crop discards along with everything a blur can carry
    # inwards from it.  Parking them one index further in -- on the ring next to
    # the image, which is where a clip to the border pixel puts them -- leaks the
    # frame's off-screen glow back into its edges, where it reads as a bright wall.
    clip_x = (float(-ring), float(scene.width + ring - 1))
    clip_y = (float(-ring), float(scene.height + ring - 1))

    z_real = np.zeros(c_real.shape, dtype=np.float64)
    z_imag = np.zeros(c_imag.shape, dtype=np.float64)
    z_real, z_imag = _iterate(z_real, z_imag, c_real, c_imag)

    for _ in range(1, scene.iterations):
        z_real, z_imag = _iterate(z_real, z_imag, c_real, c_imag)
        inside = (z_real * z_real + z_imag * z_imag) <= escape_radius_squared
        if not inside.any():
            break
        if not inside.all():
            z_real, z_imag = z_real[inside], z_imag[inside]
            c_real, c_imag = c_real[inside], c_imag[inside]
            vector_real, vector_imag = vector_real[inside], vector_imag[inside]
        pixel_x, pixel_y = scene.complex_to_pixel(z_real, z_imag)
        # Pixel x covers ``[x - 0.5, x + 0.5)``: samples are jittered around
        # the pixel centre, so the owning pixel is found by rounding, and
        # the first orbit point of a sample is then guaranteed to land back
        # on the pixel the sample came from.
        row = np.floor(np.clip(pixel_y, *clip_y) + 0.5).astype(np.int64) + ring
        column = np.floor(np.clip(pixel_x, *clip_x) + 0.5).astype(np.int64) + ring
        # Merge orbit points that share a pixel before touching the buffer:
        # a render writes to a few per cent of the image per iteration, so
        # touching every pixel would be the dominant cost by far.  The
        # indices come from `np.unique`, so the fancy-index adds below cannot
        # alias themselves, and every partial sum they perform is an exact
        # integer, which keeps the result independent of chunking.
        merged, pixel_inverse = np.unique(row * padded_width + column, return_inverse=True)
        buffer[merged, 0] += np.bincount(pixel_inverse)
        for moment, weights in ((1, vector_real), (2, vector_imag)):
            buffer[merged, moment] += np.bincount(
                pixel_inverse, weights=weights, minlength=merged.size
            )


def accumulate(
    scene: Scene, first_sample: int, last_sample: int, buffer: npt.NDArray[np.float64]
) -> None:
    """Add the orbit points of samples ``[first_sample, last_sample)`` to ``buffer``.

    ``buffer`` holds :data:`nebulabrot.scene.MOMENTS` channels over the image
    plus its discard ring.  Sample ``i`` maps to pixel ``i // samples_per_pixel``
    in row-major order.
    """
    flat = buffer.reshape(-1, MOMENTS)
    for start in range(first_sample, last_sample, CHUNK_SAMPLES):
        stop = min(start + CHUNK_SAMPLES, last_sample)
        index = np.arange(start, stop, dtype=np.uint64)
        pixel = index // scene.samples_per_pixel
        jitter_real, jitter_imag = _jitter(index, scene.seed)
        c_real, c_imag = scene.pixel_to_complex(
            (pixel % scene.width).astype(np.float64) + jitter_real,
            (pixel // scene.width).astype(np.float64) + jitter_imag,
        )
        vector_real, vector_imag = scene.flux_vector(c_real, c_imag)
        escaping = _escapes(scene, c_real, c_imag)
        _deposit_orbits(
            scene,
            flat,
            vector_real[escaping],
            vector_imag[escaping],
            c_real[escaping],
            c_imag[escaping],
        )
