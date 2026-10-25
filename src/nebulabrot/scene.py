"""Scene description: the viewport, the sampling budget and the colour ramp.

A :class:`Scene` is the single source of truth for the render.  Geometry
(complex plane <-> pixel coordinates), the escape test and the metadata that
travels inside the PNG are all derived from it.
"""

from __future__ import annotations

import dataclasses
import enum
import math

import numpy as np
import numpy.typing as npt


class Palette(enum.Enum):
    """Colour ramp: ``(age, red, green, blue)`` control points in linear light.

    ``age`` is the orbit index of a splatted point normalised to ``[0, 1]``, so
    a cloud changes hue as it ages: the youngest dust traces the viewport, while
    the oldest points lie along the rays of escaping orbits.  Points are
    interpolated linearly between control points, and every component is a
    linear-light value in ``[0, 1]``.
    """

    #: Faint indigo dust, violet mid-age, then the classic orange-gold rays.
    DEEP_FIELD = (
        (0.00, 0.010, 0.015, 0.075),
        (0.20, 0.180, 0.040, 0.380),
        (0.42, 0.720, 0.060, 0.300),
        (0.62, 0.980, 0.380, 0.040),
        (0.80, 0.980, 0.880, 0.360),
        (1.00, 0.420, 0.930, 0.760),
    )
    #: Crimson through orange to gold: one hue family, so a pixel that mixes
    #: young and old light stays inside the wedge instead of cancelling to grey.
    CRIMSON_GOLD = (
        (0.00, 0.020, 0.005, 0.040),
        (0.25, 0.330, 0.010, 0.110),
        (0.50, 0.800, 0.060, 0.020),
        (0.75, 0.990, 0.330, 0.010),
        (1.00, 0.990, 0.760, 0.060),
    )
    #: Deep blue-violet through magenta to rose, the default for the artifact.
    #:
    #: Every control point sits within about 100 degrees of hue, which is what
    #: makes a frame colourful: hue-per-age averages over the ages that reach a
    #: pixel, and averaging analogous hues leaves them coloured while averaging
    #: complementary ones (the violet-to-teal span of ``DEEP_FIELD``) walks the
    #: mixture through neutral.  Averaging also caps colour in the dense core,
    #: where most light lands: measured Oklab C* there is 0.21 for this ramp
    #: against 0.019 for ``DEEP_FIELD``.
    VIOLET_ROSE = (
        (0.00, 0.020, 0.005, 0.060),
        (0.30, 0.140, 0.010, 0.420),
        (0.60, 0.600, 0.020, 0.460),
        (1.00, 0.980, 0.100, 0.360),
    )
    #: Cold teal cloud with a hot amber core.
    EMBER = (
        (0.00, 0.010, 0.030, 0.055),
        (0.24, 0.030, 0.220, 0.360),
        (0.50, 0.420, 0.150, 0.060),
        (0.74, 0.980, 0.520, 0.070),
        (1.00, 1.000, 0.960, 0.820),
    )
    #: Violet through blue and cyan to a pale yellow eldest age.
    IRIS = (
        (0.00, 0.020, 0.010, 0.060),
        (0.26, 0.330, 0.050, 0.480),
        (0.52, 0.140, 0.300, 0.950),
        (0.78, 0.360, 0.900, 0.980),
        (1.00, 0.930, 0.990, 0.700),
    )

    @property
    def ages(self) -> tuple[float, ...]:
        """Control point positions, ascending from 0.0 to 1.0."""
        return tuple(point[0] for point in self.value)


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
    exposure: float
    density_gamma: float
    palette: Palette
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
        if self.exposure <= 0.0:
            raise ValueError(f"exposure must be positive, got {self.exposure}")
        if self.density_gamma <= 0.0:
            raise ValueError(f"density_gamma must be positive, got {self.density_gamma}")
        if not math.isfinite(self.center_real) or not math.isfinite(self.center_imag):
            raise ValueError("centre must be finite")

    @property
    def pixels_per_unit(self) -> float:
        """Resolution of the viewport; also the inverse of a pixel's width."""
        return self.width / self.span_real

    @property
    def span_imag(self) -> float:
        """Height of the viewport in complex units, set by the pixel aspect."""
        return self.height / self.pixels_per_unit

    @property
    def ring_width(self) -> int:
        """Discard border around the image, wide enough for every blur tap.

        Orbit points that leave the viewport are clamped into this ring, which
        keeps the hot loop free of masking, and the separable blur consumes one
        ring column per tap before the image is cropped.
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
        half_diagonal = 0.5 * math.hypot(self.span_real, self.span_imag)
        centre_radius = math.hypot(self.center_real, self.center_imag)
        return max(2.0, half_diagonal + centre_radius) ** 2

    @property
    def total_samples(self) -> int:
        return self.width * self.height * self.samples_per_pixel

    @property
    def accumulator_bytes(self) -> int:
        """Size of one per-worker accumulation buffer."""
        ring = self.ring_width
        return (self.height + 2 * ring) * (self.width + 2 * ring) * 3 * 8

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
        """JSON-ready description of everything that produced the image.

        The palette is named rather than inlined: its control points are part of
        the code, and the renderer version in the same chunk identifies them.
        """
        return {
            "blur_taps": self.blur_taps,
            "center_imag": self.center_imag,
            "center_real": self.center_real,
            "density_gamma": self.density_gamma,
            "exposure": self.exposure,
            "height": self.height,
            "iterations": self.iterations,
            "palette": self.palette.name,
            "samples_per_pixel": self.samples_per_pixel,
            "seed": self.seed,
            "span_imag": self.span_imag,
            "span_real": self.span_real,
            "width": self.width,
        }


#: The 4K artifact this repository is configured to produce: a wide view of the
#: set, off the real axis so the composition is not mirror-symmetric.
DEFAULT_SCENE = Scene(
    width=3840,
    height=2160,
    center_real=-1.30,
    center_imag=-0.16,
    span_real=3.0,
    iterations=500,
    samples_per_pixel=16,
    blur_taps=1,
    exposure=0.0002,
    density_gamma=0.7,
    palette=Palette.VIOLET_ROSE,
    seed=0x5EED,
)
