#!/bin/sh
# Bounded active-learning pipeline: generate simulation data in parallel, then
# train. Each phase can be run on its own so a long generation step does not
# have to be repeated to resume training.
#
#   active_train_parallel.sh generate [options]   # parallel data generation only
#   active_train_parallel.sh train    [options]   # train only, reusing existing data
#   active_train_parallel.sh all      [options]   # generate then train
#
# Every option can also be supplied as an environment variable, so a whole run
# can be configured from a single export block. Command-line options win.
#
#   VARIANT            density | density_velocity           (density)
#   DATA_DIR           session root                          (./data)
#   OUTPUT_DIR         checkpoints root                     (./attempts)
#   FRAMES_PER_RUN     frames per generated session         (100)
#   RUNS_PER_CYCLE     sessions generated per cycle         (1)
#   MAX_PARALLEL       concurrent simulator processes       (1)
#   CYCLES             generate/train cycles                (1)
#   EPOCHS             epochs per cycle                      (1)
#   BATCH_SIZE         samples per forward pass             (8)
#   EFFECTIVE_BATCH    accumulation target, >= BATCH_SIZE   (32)
#   SKIP_FRAMES        frames between context and target    (10)
#   LEARNING_RATE      AdamW learning rate                   (5e-4)
#   VALIDATION_FRACTION held-out session fraction           (0.1)
#   NUM_WORKERS        dataloader workers                   (0)
#   MODEL_FILENAME     checkpoint name                      (best_model.pth)
#   MIN_DELTA          min val-loss gain worth a 162 MiB write  (0)
#   SAVE_EVERY         only write on epochs divisible by this   (1)
#   KEEP_LAST_CHECKPOINTS  retain newest N cycle checkpoints   (0 = all)
#   DEVICE             auto | cuda | cpu                    (auto)
#   SPH_ROOT           simulator checkout                   (../sph)
#   SIM_WIDTH          resolution the simulator produces    (400)
#   SIM_HEIGHT         must match WIDTH/HEIGHT, else       (400)
#                       session validation rejects the data
#   NO_RESUME          set to restart from random init each cycle
#   PRUNE              set to trim old sessions to MAX_SESSIONS
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}

PHASE=${1:-all}
case "$PHASE" in
    generate|train|all) shift ;;
    -h|--help)
        sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        printf 'unknown phase: %s (expected generate, train, or all)\n' "$PHASE" >&2
        exit 2
        ;;
esac

require_value() {
    if [ "$#" -lt 2 ]; then
        printf 'missing value for %s\n' "$1" >&2
        exit 2
    fi
}

VARIANT=${VARIANT:-density}
DATA_DIR=${DATA_DIR:-$SCRIPT_DIR/data}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/attempts}
FRAMES_PER_RUN=${FRAMES_PER_RUN:-100}
RUNS_PER_CYCLE=${RUNS_PER_CYCLE:-1}
MAX_PARALLEL=${MAX_PARALLEL:-1}
CYCLES=${CYCLES:-1}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32}
SKIP_FRAMES=${SKIP_FRAMES:-10}
LEARNING_RATE=${LEARNING_RATE:-5e-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-0}
MODEL_FILENAME=${MODEL_FILENAME:-best_model.pth}
MIN_DELTA=${MIN_DELTA:-0}
SAVE_EVERY=${SAVE_EVERY:-1}
KEEP_LAST_CHECKPOINTS=${KEEP_LAST_CHECKPOINTS:-0}
DEVICE=${DEVICE:-auto}
MAX_SESSIONS=${MAX_SESSIONS:-0}
SIMULATION_SEED=${SIMULATION_SEED:-0}
SEED=${SEED:-0}
WIDTH=${WIDTH:-400}
HEIGHT=${HEIGHT:-400}
LATENT_DIM=${LATENT_DIM:-1024}
# Resolution the simulator will actually produce. Fixed by the C++ build; the
# model WIDTH/HEIGHT must match it or session validation will reject the data.
SIM_WIDTH=${SIM_WIDTH:-400}
SIM_HEIGHT=${SIM_HEIGHT:-400}
MAX_BATCHES=${MAX_BATCHES:-}
SMOKE=${SMOKE:-}
PRUNE=${PRUNE:-}
NO_RESUME=${NO_RESUME:-}
SPH_ROOT=${SPH_ROOT:-$SCRIPT_DIR/../sph}

# Command-line overrides for everything with an env default above.
while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)             require_value "$@"; VARIANT=$2; shift 2 ;;
        --data-dir)            require_value "$@"; DATA_DIR=$2; shift 2 ;;
        --output-dir)          require_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
        --frames-per-run)      require_value "$@"; FRAMES_PER_RUN=$2; shift 2 ;;
        --runs-per-cycle)      require_value "$@"; RUNS_PER_CYCLE=$2; shift 2 ;;
        --max-parallel)        require_value "$@"; MAX_PARALLEL=$2; shift 2 ;;
        --cycles)              require_value "$@"; CYCLES=$2; shift 2 ;;
        --epochs)              require_value "$@"; EPOCHS=$2; shift 2 ;;
        --batch-size)          require_value "$@"; BATCH_SIZE=$2; shift 2 ;;
        --effective-batch-size) require_value "$@"; EFFECTIVE_BATCH=$2; shift 2 ;;
        --skip-frames)         require_value "$@"; SKIP_FRAMES=$2; shift 2 ;;
        --learning-rate)       require_value "$@"; LEARNING_RATE=$2; shift 2 ;;
        --validation-fraction) require_value "$@"; VALIDATION_FRACTION=$2; shift 2 ;;
        --num-workers)         require_value "$@"; NUM_WORKERS=$2; shift 2 ;;
        --model-filename)      require_value "$@"; MODEL_FILENAME=$2; shift 2 ;;
        --min-delta)           require_value "$@"; MIN_DELTA=$2; shift 2 ;;
        --save-every)          require_value "$@"; SAVE_EVERY=$2; shift 2 ;;
        --keep-last-checkpoints) require_value "$@"; KEEP_LAST_CHECKPOINTS=$2; shift 2 ;;
        --device)              require_value "$@"; DEVICE=$2; shift 2 ;;
        --max-sessions)        require_value "$@"; MAX_SESSIONS=$2; shift 2 ;;
        --simulation-seed)     require_value "$@"; SIMULATION_SEED=$2; shift 2 ;;
        --seed)                require_value "$@"; SEED=$2; shift 2 ;;
        --width)               require_value "$@"; WIDTH=$2; shift 2 ;;
        --height)              require_value "$@"; HEIGHT=$2; shift 2 ;;
        --latent-dim)          require_value "$@"; LATENT_DIM=$2; shift 2 ;;
        --max-batches)         require_value "$@"; MAX_BATCHES=$2; shift 2 ;;
        --sph-root)            require_value "$@"; SPH_ROOT=$2; shift 2 ;;
        --smoke)               SMOKE=--smoke; shift ;;
        --no-resume)           NO_RESUME=1; shift ;;
        --prune)               PRUNE=--prune; shift ;;
        -h|--help)             sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Refuse the trap that fills the disk. The simulator resolution is fixed at
# compile time in ../sph/include/settings.hpp (400x400), so the budget must be
# computed from that and not from WIDTH/HEIGHT, which describe the model and
# must match the generated data. One 400x400 float32 frame is ~2.4 MiB and
# spawn_random.sh on its own defaults to 10000 frames (~24 GiB) per session.
# Create the run directories up front. DATA_DIR is frequently a fresh
# per-run path (data/run_<timestamp>) supplied by the caller, and a wrapper may
# inspect or prune it before Python starts; without this, a glob like
# "$DATA_DIR"/*/ matches nothing and cleanup reports an empty or missing
# directory instead of pruning the sessions that were just generated.
mkdir -p -- "$DATA_DIR" "$OUTPUT_DIR"

frame_bytes=$((4 * SIM_WIDTH * SIM_HEIGHT * 4))
session_bytes=$((FRAMES_PER_RUN * frame_bytes))
total_bytes=$((RUNS_PER_CYCLE * CYCLES * session_bytes))
# DATA_DIR now exists, so df measures the filesystem the sessions will land on
# rather than the nearest existing ancestor, which can be a different mount.
probe=$DATA_DIR
avail_bytes=$(df -Pk "$probe" 2>/dev/null | awk 'NR==2 {print $4 * 1024}')

if [ "$PHASE" != train ] && { [ "$WIDTH" -ne "$SIM_WIDTH" ] || [ "$HEIGHT" -ne "$SIM_HEIGHT" ]; }; then
    printf 'warning: model is %sx%s but the simulator produces %sx%s; training will reject the data\n' \
        "$WIDTH" "$HEIGHT" "$SIM_WIDTH" "$SIM_HEIGHT" >&2
    printf 'set WIDTH/HEIGHT to match, or SIM_WIDTH/SIM_HEIGHT if the C++ build was compiled differently\n' >&2
fi

if [ -z "${avail_bytes:-}" ]; then
    printf 'warning: could not determine free space for %s; skipping the budget check\n' "$probe" >&2
elif [ "$total_bytes" -gt "$avail_bytes" ]; then
    needed_gib=$(( (total_bytes / 1073741824) + 1 ))
    avail_gib=$(( avail_bytes / 1073741824 ))
    printf 'refusing to generate: %s cycles x %s runs x %s frames needs ~%s GiB but only ~%s GiB is free on %s\n' \
        "$CYCLES" "$RUNS_PER_CYCLE" "$FRAMES_PER_RUN" "$needed_gib" "$avail_gib" "$probe" >&2
    printf 'lower FRAMES_PER_RUN, RUNS_PER_CYCLE, or CYCLES\n' >&2
    exit 1
else
    printf 'data budget: %s cycles x %s runs x %s frames = ~%s MiB (%s GiB free on %s)\n' \
        "$CYCLES" "$RUNS_PER_CYCLE" "$FRAMES_PER_RUN" "$(( session_bytes / 1048576 ))" \
        "$(( avail_bytes / 1073741824 ))" "$probe"
fi

set -- \
    --phase "$PHASE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --variant "$VARIANT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --latent-dim "$LATENT_DIM" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --cycles "$CYCLES" \
    --runs-per-cycle "$RUNS_PER_CYCLE" \
    --max-parallel "$MAX_PARALLEL" \
    --max-sessions "$MAX_SESSIONS" \
    --frames-per-run "$FRAMES_PER_RUN" \
    --batch-size "$BATCH_SIZE" \
    --effective-batch-size "$EFFECTIVE_BATCH" \
    --skip-frames "$SKIP_FRAMES" \
    --learning-rate "$LEARNING_RATE" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --num-workers "$NUM_WORKERS" \
    --model-filename "$MODEL_FILENAME" \
    --min-delta "$MIN_DELTA" \
    --save-every "$SAVE_EVERY" \
    --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS" \
    --simulation-seed "$SIMULATION_SEED" \
    --seed "$SEED" \
    --sph-root "$SPH_ROOT"

if [ -n "$MAX_BATCHES" ]; then set -- "$@" --max-batches "$MAX_BATCHES"; fi
if [ -n "$SMOKE" ]; then set -- "$@" --smoke; fi
if [ -n "$PRUNE" ]; then set -- "$@" --prune; fi
if [ -n "${NO_RESUME:-}" ]; then set -- "$@" --no-resume; fi
if [ "$PHASE" = train ]; then set -- "$@" --skip-sim; fi

if [ -n "${NO_RESUME:-}" ]; then RESUME_STATE=off; else RESUME_STATE=on; fi
printf 'phase=%s variant=%s data=%s output=%s batch=%s effective=%s skip_frames=%s resume=%s\n' \
    "$PHASE" "$VARIANT" "$DATA_DIR" "$OUTPUT_DIR" "$BATCH_SIZE" "$EFFECTIVE_BATCH" \
    "$SKIP_FRAMES" "$RESUME_STATE"

exec "$PYTHON" "$SCRIPT_DIR/active_train_parallel.py" "$@"
