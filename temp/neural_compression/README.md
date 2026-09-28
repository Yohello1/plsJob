# Neural SPH Compression

This package trains and evaluates neural compression models for sessions produced by the sibling `sph` simulator. It contains the data contract, model implementations, active-learning orchestration, and command-line tools; the simulator itself remains in `../sph`.

## Architecture

- `pls_compression.schema` defines the binary frame contract, model variants, model configuration, and normalization metadata.
- `pls_compression.dataset` memory-maps each read-only session, discovers compatible sessions, and returns a previous frame plus one or more future frames. Augmentation is opt-in and is never used by evaluation tools.
- `pls_compression.models` contains the encoder, decoder, and `CompressionModel`. The encoder receives nine channels: previous density, previous velocity, target density, target velocity, obstacle mask, and two coordinate channels. The decoder predicts one density channel or density plus velocity.
- `pls_compression.training` computes statistics, performs teacher-forced and autoregressive training, validates checkpoints, and saves normalization and schema metadata with every checkpoint.
- `pls_compression.metrics` reports MSE, MAE, and structural similarity alongside the zero and identity baselines, so every number is reported next to the two trivial predictors. It adds a fluid-masked SSIM, which separates a total fluid dropout far more sharply than the unmasked mean.
- `pls_compression.evaluation` is the shared rollout and sanity-check implementation used by `check_model.py` and the tools. It performs a two-step rollout by default, uses saved normalization, and never writes to a memory map.
- `pls_compression.orchestration` is the single active-learning implementation. The shell entry points are thin wrappers around it.

## Data contract

A session is a directory containing `sim_data.bin` and preferably `metadata.json`. Every frame is a contiguous sequence of four little-endian `float32` fields:

1. `density`
2. `velocity_x`
3. `velocity_y`
4. `obstacle_mask`

The production resolution is `400 x 400`, so one frame is `4 * 400 * 400 * sizeof(float32)` bytes. The field layout is identical in both modes. Density-only still records velocity because its model consumes previous velocity as input, but predicts only the next density field. Density-plus-velocity predicts and records all three state channels.

Metadata has this form:

```json
{
  "model_variant": "density-only",
  "width": 400,
  "height": 400,
  "fields": ["density", "velocity_x", "velocity_y", "obstacle_mask"],
  "dtype": "float32"
}
```

The density-plus-velocity binary writes `"model_variant": "density-velocity"`. Python accepts either the C++ labels or the canonical `density` and `density_velocity` names and rejects sessions with a different variant.

## Modes

Density-only is the default:

```sh
python3 compressor.py --data-dir data --output-dir attempts --model-variant density
```

Density-plus-velocity is selected explicitly:

```sh
python3 compressor2.py --data-dir data --output-dir attempts
python3 train_density_velocity.py --data-dir data --output-dir attempts
```

The C++ simulation target is selected by the same variant. `spawn_random.sh` accepts `--variant density-only` or `--variant density-velocity`, builds the corresponding target when needed, runs headless, and exports an absolute `SPH_DATA_ROOT`. The legacy positional frame count remains supported:

```sh
./spawn_random.sh 250 --variant density-velocity --seed 17
```

`SPH_ROOT` or `SPH_DIR` can point to a simulator checkout. `SPH_MODEL_VARIANT`, `SPH_BUILD_TARGET`, `SPH_BINARY`, `SPH_SCENARIO_SEED`, `SPH_DATA_ROOT`, `FRAMES_PER_RUN`, and `MAKE` provide equivalent environment controls. Scenario arguments are deterministic for a given seed.

## Build and smoke commands

Install the Python dependencies with:

```sh
python3 -m pip install -r requirements.txt
```

Build the simulator variants from this directory with:

```sh
make -C ../sph draw2-density-only
make -C ../sph draw2-density-velocity
make -C ../sph check
```

Run the simulator smoke check with:

```sh
make -C ../sph smoke
```

The simulator build uses AdaptiveCpp (`acpp`) and the target can be selected with `ACPP_TARGETS`, for example `make -C ../sph ACPP_TARGETS=cuda draw2-density-velocity`. SDL is optional; headless operation does not require a display.

## Training and evaluation

A CPU smoke training run with existing data is:

```sh
python3 train_density_only.py --data-dir data --output-dir attempts --smoke --device cpu
python3 train_density_velocity.py --data-dir data --output-dir attempts --smoke --device cpu
```

The production startup smoke generates two native frames and executes one real 400×400 forward/backward batch without completing an epoch. It selects CUDA when available and otherwise uses CPU:

```sh
make smoke
python3 smoke_training.py --variant density --device auto
python3 smoke_training.py --variant density_velocity --device auto
```

The richer checker reports per-step MSE, zero baseline, identity baseline, and SSIM, and uses the checkpoint's saved normalization and model configuration:

```sh
python3 check_model.py --model attempts/best_model.pth --data-dir data --steps 2
python3 -m pls_compression check --model attempts/best_model.pth --data-dir data --steps 2
```

The old visualization intents are available as tools rather than executable tests:

```sh
python3 tools/reconstruct.py --model attempts/best_model.pth --data-dir data --plot --out reconstruction.png
python3 tools/iterative.py --model attempts/best_model.pth --data-dir data --steps 2 --plot --out stability.png
```

Matplotlib is imported only when `--plot` is requested. Numerical checks and two-step rollouts work without it. DataParallel-prefixed checkpoints are accepted.

## Structural similarity

`pls_compression.metrics` implements SSIM with the standard 11x11 Gaussian window, `sigma=1.5`, and a data range of 1.0, which matches both the sigmoid density range and the tanh velocity range. It is available as a function, as a per-pixel map, and inside `compute_metrics`:

```python
from pls_compression.metrics import ssim, ssim_map, ssim_fluid, compute_metrics

ssim(prediction, target)             # scalar mean structural similarity
ssim_map(prediction, target)         # [..., height, width] map, channels averaged
ssim_fluid(prediction, target)       # scalar over the fluid region only
ssim(prediction[:, 0:1], target[:, 0:1])   # density channel only
compute_metrics(prediction, target)  # adds "ssim", "ssim_density", "ssim_fluid"
```

**Prefer `ssim_fluid` on this data.** Measured on a real 400x400 session, comparing each prediction against the next frame:

| prediction | `ssim` | `ssim_fluid` (0.05) | `ssim_fluid` (0.10) |
| --- | --- | --- | --- |
| perfect | 1.0000 | 1.0000 | 1.0000 |
| zero predictor | 0.5025 | 0.0069 | 0.0000 |
| identity (copy previous frame) | 0.9731 | 0.9523 | 0.5524 |
| fluid region zeroed | 0.5028 | 0.0069 | -0.0001 |
| 9x9 box-blurred | 0.9326 | 0.9219 | 0.6284 |

The unmasked mean does react to a total dropout, but it only falls to 0.50, which leaves it barely distinguishable from a moderately bad prediction. The masked score separates the same two cases by 0.0069 against 1.0, so a model that drops fluid is unambiguously worse than one that does not.

The threshold column matters too. At 0.05 roughly 47% of a normalized frame is above the threshold, because SPH density has a diffuse halo around the particles; tightening to 0.10 selects the 5% densest core. `ssim_fluid` therefore shifts with the threshold, and the identity baseline drops from 0.95 to 0.55, which is the intended behaviour: the copy-previous-frame predictor is excellent over the halo and much worse over the dense core. Pick the threshold to match what you care about and keep it fixed across comparisons, because the score is not comparable between thresholds.

`padding="same"` (the default) keeps the map aligned with the input image. Pass `padding="valid"` to crop the border and match reference implementations that score only fully covered windows; the two differ in the outer few pixels, so comparisons against other tools should use the same mode. Channels are averaged, so slice the density channel out to score it alone. A window larger than the image is reduced to the largest odd size that fits, and images smaller than 3x3 fall back to global statistics instead of failing.

All three values are computed from a single convolution pass, so `compute_metrics` pays for one SSIM, not three. They run each epoch during validation, and `losses.csv` records them as `val_ssim`, `val_ssim_density`, and `val_ssim_fluid`. SSIM is a measurement only: it is never part of the loss, so these remain independent of what training optimizes.

## Tests

Run all Python tests with the standard library runner:

```sh
python3 -m compileall -q .
python3 -m unittest discover -v
```

The tests use `unittest` only. Tests that need real simulator binaries are skipped when those binaries have not been built.

## Active learning

The loop resolves the simulator relative to this checkout, exports `SPH_DATA_ROOT` for each run, passes deterministic seeds, and propagates simulator or training failures. It does not prune data by default. Use `--prune` only when old sessions should be removed, and set `--max-sessions` explicitly. The `dry_run` argument to `cleanup_storage` provides a non-destructive pruning preview.

### Phases

`--phase` selects what a cycle does:

| phase | generates data | trains |
| --- | --- | --- |
| `all` (default) | yes | yes |
| `generate` | yes | no |
| `train` | no | yes |

Splitting them means a long generation step does not have to be repeated to train again, and lets you review the data before spending GPU time. `--skip-sim` is shorthand for `--phase train`; combining it with `--phase generate` is an error because nothing would run.

### Resuming across cycles

By default each cycle starts from the previous cycle's best checkpoint, so cycles accumulate instead of restarting from a random initialisation. Measured across two cycles on 400x400 data, the cycle 2 weights sit 0.039 relative distance from cycle 1 but 1.42 from a fresh model, confirming the transfer. Pass `--no-resume` to restart from scratch each cycle, which is what you want when cycles are meant to be independent comparisons. A missing or unwritten checkpoint is not an error; the chain simply starts fresh.

Checkpoints are written per cycle to `<output-dir>/cycle_N/<model-filename>`.

### Training flags

`active_train_parallel.sh` and both Python entry points accept the full training configuration, not just the loop limits:

| flag | default | note |
| --- | --- | --- |
| `--batch-size` | 8 | samples per forward pass |
| `--effective-batch-size` | 32 | accumulation target, must be at least `--batch-size` |
| `--skip-frames` | 10 | frames between context and target |
| `--n-steps` | 1 | autoregressive rollout steps per sample |
| `--skip-initial` | 1 | stride over candidate samples |
| `--learning-rate` | 5e-4 | |
| `--validation-fraction` | 0.1 | held-out session fraction |
| `--gradient-clip-norm` | unset | |
| `--noise-std` | 0.0 | input noise during training |
| `--num-workers` | 0 | dataloader workers |
| `--model-filename` | `best_model.pth` | |
| `--no-resume` | off | restart from random init each cycle |

Model shape flags (`--base-channels`, `--bottleneck-channels`, `--context-channels`, `--projection-dim`, `--num-downsamples`) are exposed alongside them, matching `pls-compression train`. Invalid combinations, such as an effective batch below the batch size or a validation fraction outside `(0, 1)`, are rejected when the configuration is built rather than at the first training step.

### The pipeline script

`active_train_parallel.sh` is a thin POSIX shell wrapper that takes a phase and then forwards options to the Python entry point. Every option has an environment-variable default, so a whole run can be configured from one export block, and command-line options override the environment.

```sh
# 1. generate training data with four simulators in parallel
FRAMES_PER_RUN=100 RUNS_PER_CYCLE=8 MAX_PARALLEL=4 \
  ./active_train_parallel.sh generate

# 2. train against it, resuming from the previous cycle if one exists
SKIP_FRAMES=1 BATCH_SIZE=8 EFFECTIVE_BATCH=32 EPOCHS=100 DEVICE=cuda CYCLES=1 \
  ./active_train_parallel.sh train

# or both in one call
./active_train_parallel.sh all --variant density_velocity --epochs 50
```

Environment variables: `VARIANT`, `DATA_DIR`, `OUTPUT_DIR`, `FRAMES_PER_RUN`, `RUNS_PER_CYCLE`, `MAX_PARALLEL`, `CYCLES`, `EPOCHS`, `BATCH_SIZE`, `EFFECTIVE_BATCH`, `SKIP_FRAMES`, `LEARNING_RATE`, `VALIDATION_FRACTION`, `NUM_WORKERS`, `MODEL_FILENAME`, `DEVICE`, `MAX_SESSIONS`, `SIMULATION_SEED`, `SEED`, `WIDTH`, `HEIGHT`, `LATENT_DIM`, `MAX_BATCHES`, `SMOKE`, `SPH_ROOT`, `NO_RESUME`, `PRUNE`. Run `./active_train_parallel.sh --help` for the full list.

Before generating anything the script computes the data budget from `frames_per_run`, `runs_per_cycle`, and `cycles`, and refuses to start when the projected size exceeds the free space on the target filesystem. This matters because `spawn_random.sh` on its own defaults to 10000 frames per session, which is roughly 24 GiB at 400x400. The check resolves the nearest existing ancestor directory, since the data directory usually does not exist yet, and warns rather than skipping if free space cannot be determined.

`active_train.sh` remains a bare wrapper around the same Python entry point. It accepts every flag listed above but does not have the phase parsing, the environment defaults, or the data budget check.

### Legacy entry points

```sh
./active_train.sh --data-dir data --output-dir attempts --model-variant density --runs-per-cycle 1
SPH_MODEL_VARIANT=density_velocity ./active_train.sh --data-dir data --output-dir attempts --runs-per-cycle 1
```
