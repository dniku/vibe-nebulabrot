"""Render a scene, in parallel, and write the PNG artifact.

Every worker owns a full-size accumulation buffer rather than a strip of one:
an orbit point can land anywhere in the viewport, so a strip would silently lose
the contributions of samples owned by other workers, which is exactly what makes
the long rays of a Nebulabrot reach across the frame.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import multiprocessing
import os
import pathlib
import sys
import time

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.PngImagePlugin

from nebulabrot import __version__, colour, kernel
from nebulabrot.scene import Scene


def default_jobs(scene: Scene) -> int:
    """Worker count that keeps one accumulation buffer per worker in RAM.

    A 4K buffer is about 200 MB, so on a machine with less free memory than
    cores it is better to use fewer workers than to start swapping.
    """
    cores = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as meminfo:
            available = next(
                int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:")
            )
    except OSError, StopIteration:
        return cores
    # Leave half of the available memory to everything else on the machine.
    return max(1, min(cores, int(available / 2) // scene.accumulator_bytes))


def _render_samples(scene: Scene, first_sample: int, last_sample: int) -> npt.NDArray[np.float64]:
    ring = scene.ring_width
    buffer = np.zeros((scene.height + 2 * ring, scene.width + 2 * ring, 3), dtype=np.float64)
    kernel.accumulate(scene, first_sample, last_sample, buffer)
    return buffer


def _sample_bounds(scene: Scene, jobs: int) -> list[tuple[int, int]]:
    total = scene.total_samples
    bounds = [job * total // jobs for job in range(jobs + 1)]
    return list(itertools.pairwise(bounds))


def accumulate(scene: Scene, jobs: int) -> npt.NDArray[np.float64]:
    """Accumulate every sample of ``scene`` in linear light, ring already cropped.

    The result still has to be tone mapped, which is a separate decision from
    how much light is there: ``--exposure`` can be retuned without re-rendering.
    """
    started = time.monotonic()
    ring = scene.ring_width
    accumulated = np.zeros((scene.height + 2 * ring, scene.width + 2 * ring, 3), dtype=np.float64)
    bounds = _sample_bounds(scene, jobs)
    if jobs == 1:
        for first, last in bounds:
            accumulated += _render_samples(scene, first, last)
    else:
        # `fork` rather than the 3.14 default `forkserver`: the parent has no
        # threads and no locks to inherit, and a fork does not need to bind a
        # socket, which keeps the renderer usable inside sandboxes.
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            results = pool.map(
                _render_samples,
                [scene] * len(bounds),
                [first for first, _ in bounds],
                [last for _, last in bounds],
                chunksize=1,
                # Accumulate as bands finish instead of buffering 200 MB each.
                buffersize=1,
            )
            for done, partial in enumerate(results, start=1):
                accumulated += partial
                print(
                    f"[{time.monotonic() - started:6.1f}s] {done}/{jobs} sample bands",
                    file=sys.stderr,
                    flush=True,
                )
    return colour.smooth(accumulated, taps=scene.blur_taps, ring=ring)


def render(scene: Scene, jobs: int) -> npt.NDArray[np.uint8]:
    """Render ``scene`` to 8-bit sRGB."""
    return colour.to_srgb(accumulate(scene, jobs), scene.exposure)


def write_png(image: npt.NDArray[np.uint8], scene: Scene, path: pathlib.Path) -> None:
    """Write ``image`` deterministically: no timestamps, fixed chunk order."""
    metadata = PIL.PngImagePlugin.PngInfo()
    metadata.add_text("Software", f"nebulabrot {__version__}")
    metadata.add_text(
        "Description",
        json.dumps(
            {
                "renderer": __version__,
                "scene": scene.metadata(),
            },
            sort_keys=True,
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(image).save(path, format="PNG", compress_level=9, pnginfo=metadata)
