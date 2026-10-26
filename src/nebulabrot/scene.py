"""Scene description: the viewport, the sampling budget and the colour geometry.

A :class:`Scene` is the single source of truth for the render.  Geometry
(complex plane <-> pixel coordinates), the escape test, the vector each orbit
point carries and the metadata that travels inside the PNG are all derived from
it.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt

#: Accumulation channels: orbit count, then the real and imaginary parts of the
#: accumulated flux.  See :meth:`Scene.flux_vector`.
MOMENTS = 3


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scene:
    """Everything that decides what the rendered pixels look like."""

    width: int
    height: int
    center_real: float
    center_imag: float
    span_real: float
    iterations: int
    samples_per_pixel: int
    blur_taps: int
    #: Orbit points in a pixel that put it at half light.  Scales with the
    #: sampling budget, like any exposure.
    tone_mid: float
    #: Powers of ten of orbit density, either side of :attr:`tone_mid`, that
    #: span black to white.  Half of it: the window is symmetric in log density.
    tone_decades: float
    #: Lightness of no light at all.  It is an asymptote of the tone curve rather
    #: than a floor it is clipped to, so the density gradient keeps running all
    #: the way down into it.  Zero sends the corners of the frame to black, which
    #: puts the nebula's own falloff on show as the edge of a disc; a small
    #: non-zero value lands the falloff on a deep coloured field instead, which is
    #: where the hue the flux measured still shows.
    tone_floor: float
    #: How much a pixel that has lost its polarization is dimmed, in density.
    #: Light that arrives from every direction has nowhere interesting to be said
    #: to come from, and dimming it is what lets the interior of the set read as a
    #: silhouette and the dust lanes read as dark.  Measured over
    #: :attr:`shadow_neighbourhood`.
    coherence_shadow: float
    #: How much of a pixel's lightness is spent on colour.  A fully polarized
    #: pixel ends up at ``chroma_gain`` times its lightness.
    chroma_gain: float
    #: Exponent of the polarization in the chroma.  Below 1 it keeps colour in the
    #: softly polarized middle of the frame as well as in the beams.
    chroma_gate: float
    #: Radius, in pixels, of the neighbourhood a pixel's colour is judged
    #: against; 0 compares it against nothing and lets sampling noise through
    #: as coloured grain.
    chroma_neighbourhood: int
    #: Radius, in pixels, of the neighbourhood the :attr:`coherence_shadow` is
    #: measured over.  Much smaller than :attr:`chroma_neighbourhood`: a shadow
    #: has to keep the fine structure the orbit count carries, and unlike a hue
    #: it can afford to be a little noisy, since it only dims.
    shadow_neighbourhood: int
    #: Real part of the pole the hue wheel turns about; see
    #: :meth:`flux_vector`.
    anchor_real: float
    #: Imaginary part of the pole.
    anchor_imag: float
    #: How many times the circle of seed directions is folded before they are
    #: averaged: 1 asks which side of the pole the light came from, 2 which
    #: line through it.  See :meth:`flux_vector`.
    harmonic: int
    #: Rotation in radians of the whole palette about the hue wheel.
    hue_offset: float
    #: How much the palette lingers on the hues around blue-violet, spending more
    #: of the frame on them and hurrying through the warm end instead.  See
    #: :func:`nebulabrot.colour.palette_hue`.
    hue_dwell: float
    seed: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"image must be at least 1x1, got {self.width}x{self.height}")
        if self.span_real <= 0.0:
            raise ValueError(f"span_real must be positive, got {self.span_real}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.samples_per_pixel < 1:
            raise ValueError(f"samples_per_pixel must be positive, got {self.samples_per_pixel}")
        if self.blur_taps < 0:
            raise ValueError(f"blur_taps must not be negative, got {self.blur_taps}")
        if self.tone_mid <= 0.0:
            raise ValueError(f"tone_mid must be positive, got {self.tone_mid}")
        if self.tone_decades <= 0.0:
            raise ValueError(f"tone_decades must be positive, got {self.tone_decades}")
        if not 0.0 <= self.tone_floor < 1.0:
            raise ValueError(f"tone_floor must be in [0, 1), got {self.tone_floor}")
        if self.chroma_gain < 0.0:
            raise ValueError(f"chroma_gain must not be negative, got {self.chroma_gain}")
        if self.shadow_neighbourhood < 0:
            raise ValueError(
                f"shadow_neighbourhood must not be negative, got {self.shadow_neighbourhood}"
            )
        if self.chroma_neighbourhood < 0:
            raise ValueError(
                f"chroma_neighbourhood must not be negative, got {self.chroma_neighbourhood}"
            )
        if not math.isfinite(self.center_real) or not math.isfinite(self.center_imag):
            raise ValueError("centre must be finite")
        if not math.isfinite(self.anchor_real) or not math.isfinite(self.anchor_imag):
            raise ValueError("anchor must be finite")
        if self.harmonic not in (1, 2):
            raise ValueError(f"harmonic must be 1 or 2, got {self.harmonic}")
        if self.coherence_shadow < 0.0:
            raise ValueError(f"coherence_shadow must not be negative, got {self.coherence_shadow}")
        if not 0.0 <= self.chroma_gate <= 1.0:
            raise ValueError(f"chroma_gate must be in [0, 1], got {self.chroma_gate}")
        if abs(self.hue_dwell) > 1.0:
            raise ValueError(
                f"hue_dwell must be in [-1, 1] to keep the palette monotone, got {self.hue_dwell}"
            )

    @property
    def pixels_per_unit(self) -> float:
        """Resolution of the viewport; also the inverse of a pixel's width."""
        return self.width / self.span_real

    @property
    def half_diagonal(self) -> float:
        """Half the viewport diagonal, in complex units."""
        return 0.5 * math.hypot(self.span_real, self.span_imag)

    @property
    def span_imag(self) -> float:
        """Height of the viewport in complex units, set by the pixel aspect."""
        return self.height / self.pixels_per_unit

    @property
    def ring_width(self) -> int:
        """Discard border around the image, wide enough for every blur tap.

        Orbit points that leave the viewport are clamped into this ring, which
        keeps the hot loop free of masking, and the splat blur consumes one ring
        column per tap before the image is cropped.
        """
        return self.blur_taps + 1

    @property
    def escape_radius_squared(self) -> float:
        """Squared radius at which an orbit may be abandoned.

        An orbit point that lands inside the viewport satisfies
        ``|z| <= |centre| + half diagonal``.  Once ``|z| > 2`` the iteration
        diverges monotonically, so abandoning at the larger of the two bounds
        cannot discard a point that would have been drawn.
        """
        centre_radius = math.hypot(self.center_real, self.center_imag)
        return max(2.0, self.half_diagonal + centre_radius) ** 2

    @property
    def total_samples(self) -> int:
        return self.width * self.height * self.samples_per_pixel

    @property
    def accumulator_bytes(self) -> int:
        """Size of one per-worker accumulation buffer."""
        ring = self.ring_width
        return (self.height + 2 * ring) * (self.width + 2 * ring) * MOMENTS * 8

    def flux_vector(
        self, real: npt.NDArray[np.float64], imag: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Unit vector each seed carries, quantized onto the accumulation grid.

        A Nebulabrot has one number per pixel and one number cannot carry hue:
        any ramp over it paints the frame in a single colour that only varies in
        brightness.  What carries hue here is *where the light came from*.  Each
        orbit point carries a unit vector pointing from the pole --
        ``anchor_real + anchor_imag i`` -- to the seed that produced it, and a
        pixel accumulates the vectors of every orbit point that lands on it.  The
        direction of the sum is that mixture's circular mean: a mean over the
        many seeds whose orbits cross the pixel, so it varies smoothly across the
        frame, and it is nothing to do with how bright the pixel is.

        Folding the direction ``harmonic`` times before averaging is what makes
        the mixture cancel in interesting places.  At 1 the vectors of two seeds
        on opposite sides of the pole cancel outright, which puts the neutral
        line through the pole; at 2 they agree instead, and the cancellation
        happens along the pair of curves where the seeds' directions are a quarter
        turn apart.

        A seed sitting on the pole has no direction, and carries no vector; it
        still counts towards the pixel's density, so it dilutes the mixture the
        way an unpolarized contribution would.
        """
        offset_real = real - self.anchor_real
        offset_imag = imag - self.anchor_imag
        length = np.hypot(offset_real, offset_imag)
        unit_real = np.divide(offset_real, length, out=np.zeros_like(length), where=length > 0.0)
        unit_imag = np.divide(offset_imag, length, out=np.zeros_like(length), where=length > 0.0)
        if self.harmonic == 2:
            unit_real, unit_imag = (
                unit_real * unit_real - unit_imag * unit_imag,
                2.0 * unit_real * unit_imag,
            )
        return np.rint(unit_real / FLUX_QUANTUM), np.rint(unit_imag / FLUX_QUANTUM)

    def pixel_to_complex(
        self, pixel_x: npt.NDArray[np.float64], pixel_y: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map sub-pixel coordinates (pixel centres at integers) to ``c``."""
        return (
            self.center_real + (pixel_x - (self.width - 1) / 2) / self.pixels_per_unit,
            self.center_imag - (pixel_y - (self.height - 1) / 2) / self.pixels_per_unit,
        )

    def complex_to_pixel(
        self, real: npt.NDArray[np.float64], imag: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Inverse of :meth:`pixel_to_complex`."""
        return (
            (real - self.center_real) * self.pixels_per_unit + (self.width - 1) / 2,
            (self.center_imag - imag) * self.pixels_per_unit + (self.height - 1) / 2,
        )

    def metadata(self) -> dict[str, object]:
        """JSON-ready description of everything that produced the image."""
        return {
            "anchor_imag": self.anchor_imag,
            "anchor_real": self.anchor_real,
            "blur_taps": self.blur_taps,
            "center_imag": self.center_imag,
            "center_real": self.center_real,
            "hue_dwell": self.hue_dwell,
            "hue_offset": self.hue_offset,
            "chroma_gain": self.chroma_gain,
            "chroma_gate": self.chroma_gate,
            "chroma_neighbourhood": self.chroma_neighbourhood,
            "coherence_shadow": self.coherence_shadow,
            "shadow_neighbourhood": self.shadow_neighbourhood,
            "harmonic": self.harmonic,
            "height": self.height,
            "iterations": self.iterations,
            "samples_per_pixel": self.samples_per_pixel,
            "seed": self.seed,
            "tone_decades": self.tone_decades,
            "tone_floor": self.tone_floor,
            "tone_mid": self.tone_mid,
            "width": self.width,
        }


#: Quantum of the accumulated flux vector.
#:
#: Deposits are integers so that every partial sum the renderer makes is exact
#: and the result cannot depend on the order the sums are taken in -- which is
#: what makes the picture bit-identical whatever ``--jobs`` is.  The bound is the
#: pathological one, all of a 4K render's orbit points landing in one pixel:
#: 3840*2160*32*500 = 1.3e11 deposits, each at most ``1 / FLUX_QUANTUM`` = 8192
#: in magnitude, giving 1.1e15 against the 9.0e15 that float64 adds exactly.
FLUX_QUANTUM = 2.0**-13

#: The 4K artifact this repository is configured to produce: the whole set, framed
#: so that its silhouette fills the frame, with the hue pole on the viewport
#: centre so the palette turns about the subject.
#:
#: ``tone_mid`` belongs to this sampling budget rather than to the geometry: orbit
#: counts scale with ``samples_per_pixel``, so it is the median density of the
#: 3840x2160 / 32-sample / 500-iteration field once the coherence shadow is on it,
#: and it has to move with any of those three.
#:
#: The grade is the one that was measured to sit nearest the Nylander reference's
#: tone statistics while carrying more colour: lightness 5th/50th/95th percentile
#: 0.09/0.39/0.82 and chroma-to-lightness 0.22, against 0.17/0.33/0.63 and 0.176
#: there.
DEFAULT_SCENE = Scene(
    width=3840,
    height=2160,
    center_real=-0.80,
    center_imag=0.0,
    span_real=2.5,
    iterations=500,
    samples_per_pixel=32,
    blur_taps=2,
    tone_mid=64.0,
    tone_decades=1.15,
    tone_floor=0.05,
    coherence_shadow=0.85,
    chroma_gain=0.6,
    chroma_gate=0.7,
    chroma_neighbourhood=72,
    shadow_neighbourhood=16,
    anchor_real=-0.80,
    anchor_imag=0.0,
    harmonic=1,
    hue_offset=-0.20943951023931956,  # -12 degrees
    hue_dwell=0.3,
    seed=0x5EED,
)
