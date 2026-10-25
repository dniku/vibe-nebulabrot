"""Orbit accumulation: the compute-bound inner loop of the renderer.

A Nebulabrot plots orbits instead of escaping them.  Every sample is a point
``c`` of the Mandelbrot parameter plane; its orbit ``z <- z^2 + c`` is iterated
and each ``z`` drops light into whichever pixel of the *same* viewport it
lands on.  Dense, slowly-moving orbits build glowing clouds, and orbits that
escape paint the rays that shoot across the frame.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nebulabrot.colour import iteration_weights
from nebulabrot.scene import Scene

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


def accumulate(
    scene: Scene, first_sample: int, last_sample: int, buffer: npt.NDArray[np.float64]
) -> None:
    """Add the orbit points of samples ``[first_sample, last_sample)`` to ``buffer``.

    ``buffer`` covers the image plus a discard ring of ``scene.ring_width``
    pixels; orbit points that leave the viewport are clamped into the ring,
    which is dropped by :func:`nebulabrot.colour.smooth`.  Sample ``i`` maps to
    pixel ``i // samples_per_pixel`` in row-major order.
    """
    weights = iteration_weights(scene.palette, scene.iterations)
    escape_radius_squared = scene.escape_radius_squared
    ring = scene.ring_width
    padded_width = scene.width + 2 * ring
    # Clamp bounds: after adding the ring these stay inside `buffer`.
    clip_x = (-1.0, float(scene.width))
    clip_y = (-1.0, float(scene.height))
    flat_buffer = buffer.reshape(-1, 3)

    for start in range(first_sample, last_sample, CHUNK_SAMPLES):
        stop = min(start + CHUNK_SAMPLES, last_sample)
        index = np.arange(start, stop, dtype=np.uint64)
        pixel = index // scene.samples_per_pixel
        jitter_real, jitter_imag = _jitter(index, scene.seed)
        c_real, c_imag = scene.pixel_to_complex(
            (pixel % scene.width).astype(np.float64) + jitter_real,
            (pixel // scene.width).astype(np.float64) + jitter_imag,
        )
        z_real = np.zeros(c_real.shape, dtype=np.float64)
        z_imag = np.zeros(c_imag.shape, dtype=np.float64)

        for iteration in range(scene.iterations):
            next_real = z_real * z_real - z_imag * z_imag + c_real
            z_imag = z_real * z_imag + z_real * z_imag + c_imag
            z_real = next_real
            # False for NaN and infinity too, so diverged samples die here.
            inside = (z_real * z_real + z_imag * z_imag) <= escape_radius_squared
            if not inside.any():
                break
            if not inside.all():
                z_real, z_imag = z_real[inside], z_imag[inside]
                c_real, c_imag = c_real[inside], c_imag[inside]
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
            # indices come from `np.unique`, so the fancy-index add below cannot
            # alias itself, and every partial sum it performs is an exact
            # integer, which keeps the result independent of chunking.
            pixel_index, counts = np.unique(row * padded_width + column, return_counts=True)
            flat_buffer[pixel_index] += counts[:, None] * weights[iteration]
