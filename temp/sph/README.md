# SPH simulator

The simulator uses AdaptiveCpp SYCL queues and SDL2 when SDL2 is available. The default build is the density-only training variant.

## Build

```sh
make
make draw2-density-only
make draw2-density-velocity
```

`make` creates `draw2` as the compatibility path for `draw2-density-only`. The two variants use separate object directories and cannot share compiled macro state. The AdaptiveCpp backend defaults to OpenMP and can be selected with `ACPP_TARGETS`, for example `make ACPP_TARGETS=cuda:sm_89`.

Run the pure C++ checks with:

```sh
make check
```

Run two-frame headless smoke tests for both variants with:

```sh
make smoke
```

## Usage

```text
draw2 [frame_count] [options]
draw2 --frames frame_count [options]
```

Options are `--headless`, `--render particles|density`, `--render-density`, `--fluid x y width height`, and `--ghost x y width height`. The `-f` and `-g` aliases are also accepted. Frame counts must be positive; the default is one frame. The old `SPH_DATA_ROOT`, `sim_data.bin`, and `draw2` interfaces are retained.

Particle rendering is the default. Density rendering uses the same rasterized density field written to `sim_data.bin`, with a square-root-scaled black-to-blue-to-white heatmap:

```sh
./draw2 1000 --render density --fluid 120 120 60 60
./draw2 1000 --render-density --fluid 120 120 60 60
```

## Data sessions

Each run creates a timestamped directory below `SPH_DATA_ROOT` (or `data` when the variable is unset). The directory contains `sim_data.bin` and `metadata.json`. Every frame contains four contiguous float32 fields in this order:

1. density
2. velocity_x
3. velocity_y
4. obstacle_mask

The frame size is `width * height * 4 * sizeof(float)`. Both variants preserve the same four-field layout and record velocity because the density-only model also consumes previous velocity as input.
