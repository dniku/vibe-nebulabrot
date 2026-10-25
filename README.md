# nebulabrot

A renderer for the **Nebulabrot**: instead of asking whether a Mandelbrot orbit escapes, plot the orbit itself. Every pixel of the frame is a seed `c`, its orbit `z <- z^2 + c` is iterated, and each `z` deposits colour into whichever pixel of the *same* frame it lands on. Dense, slow-moving orbits build nebula clouds; escaping orbits streak across the frame as rays. The frame is therefore not a picture of the parameter plane, it is a picture of where orbits of the parameter plane go.

The render is bit-reproducible: the same commit, the same flags and the pinned Nix inputs produce identical PNG bytes on any machine and for any value of `--jobs`.

## Usage

```console
$ nix build .#image            # render the 4K artifact to ./result
$ nix build .#image --print-out-paths
/nix/store/…-nebulabrot-3840x2160.png
$ nix run .#nebulabrot -- --output nebula.png --iterations 200 --exposure 0.0004
$ nix develop                  # renderer, Python env, ruff, pyrefly
$ nix build .#nebulabrot        # runs ruff format/check, pyrefly and pytest
```

Rendering 3840×2160 at 16 samples per pixel and 500 iterations takes about three minutes, and each worker holds roughly 200 MB of accumulator, so `--jobs` defaults to a count derived from the core count and `MemAvailable` rather than from cores alone.

Adding workers buys little past a handful: on a 24-core host `--jobs` 2, 8 and 24 take 173 s, 173 s and 168 s. Splats land at effectively random addresses in a 199 MB accumulator (3840·2160·3 channels·8 bytes), so the render waits on memory traffic rather than on cores. `--jobs` is a memory knob here, not a speed knob.

## How a frame is built

1. **Sample** (`src/nebulabrot/kernel.py`). Each pixel contributes `--samples` seeds, jittered inside the pixel by a SplitMix64 hash of `(seed, sample index)`. Being a hash of the sample index rather than a pull from a random stream means the offsets do not depend on how samples are split between chunks or workers.
2. **Iterate and splat** (`src/nebulabrot/kernel.py`). Orbits are advanced a whole chunk at a time; samples whose `|z|` exceeds `Scene.escape_radius_squared` are dropped, which is safe because past `|z| > 2` an orbit diverges monotonically and can never return to the frame. Orbit points that share a pixel are merged before touching the accumulator: only a few per cent of the image is written in a given iteration, so a dense per-iteration update would be the dominant cost by far.
3. **Accumulate colour by orbit age** (`src/nebulabrot/colour.py`). A splat deposits the palette colour of its orbit index, so hue reports how far a point has travelled from its seed. Weights are rounded to multiples of `1/65536`, which keeps every partial sum an exact integer: the accumulation is then genuinely associative, and that is the property `--jobs`-independence rests on.
4. **Smooth** (`nebulabrot.colour.smooth`). Each orbit point is spread over the two pixels around its sub-pixel position, a 2-tap tent per axis. Because accumulation is linear, applying that tent once to the summed buffer equals applying it to every splat, at a fraction of the cost. `--blur-taps 1` is exactly bilinear splatting; more taps give a softer nebula.
5. **Tone map** (`nebulabrot.colour.to_srgb`). Orbit density is warped by `--density-gamma`, scaled by `--exposure`, compressed with Reinhard (`x / (1 + x)`, which saturates towards white like a long exposure) and encoded through the sRGB transfer function. The warp matters more than it looks: orbit density spans about five decades between the faint dust and the core of the nebula, and without it the frame is a bright blob floating in flat black — at `--density-gamma 1` the default scene is 64 % genuinely black and shows nothing between 0 and 200.

Each worker owns a full-frame accumulator rather than a strip of one, because an orbit point can land anywhere in the frame: strips would silently drop the contributions of samples owned by other workers, which is exactly what makes the rays reach across the image.

## Reproducibility

- **Inputs.** `flake.lock` pins nixpkgs to one revision; numpy 2.5.2 and Pillow 12.3.0 come from there.
- **Arithmetic.** Accumulated values are exact integer sums, so neither chunking nor process count changes a single bit (`tests/test_kernel.py::test_partitioning_samples_does_not_change_the_image`).
- **Randomness.** Jitter is a hash of the sample index and `--seed`, never a stream position.
- **Container.** The PNG carries the scene parameters as a `tEXt` chunk, is written without timestamps, and uses a fixed zlib level, so the file has nothing in it that varies between runs.
- **Measured.** The shipped frame re-rendered at `--jobs` 8 and 24, and built in the Nix sandbox at `--jobs` 2 (`NIX_BUILD_CORES`), all give `sha256:a2e6e5b56274b60fa31245c85ff204978532cfc14d84af3b5f130f3337579e34`.

To confirm it yourself, render the same scene with different parallelism and compare:

```console
$ nix run .#nebulabrot -- --jobs 1 --output /tmp/a.png
$ nix run .#nebulabrot -- --jobs 8 --output /tmp/b.png
$ sha256sum /tmp/a.png /tmp/b.png   # identical
```

## Tuning a frame

`--center-real`, `--center-imag` and `--span-real` choose the viewport in the parameter plane; `--iterations` sets how long orbits live and therefore how far the rays reach and how much of the colour ramp is used; `--samples` trades noise for time; `--density-gamma` decides how the decades of density are shared out; `--exposure` sets where the Reinhard knee sits, in units of `COLOUR_LEVELS / exposure` of accumulated light. The defaults in `nebulabrot.scene.DEFAULT_SCENE` are the artifact `nix build .#image` produces. A wide view placed off the real axis avoids a mirror-symmetric composition.

## Residual notes

- The renderer holds one full-frame `float64` accumulator per worker, so peak memory is `jobs × width × height × 3 × 8` bytes; it does not stream or tile.
- Orbits are computed in `float64`, which is stable for the iteration counts used here but is not a proof that a plotted point is on the true orbit of its seed: chaotic orbits diverge from the exact trajectory long before the iteration budget ends. Nebulabrot renders a density field, so that difference is not visible, but the image is best described as a picture of this computation.
