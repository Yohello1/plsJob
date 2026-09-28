#!/usr/bin/env bash
set -euo pipefail

script_path=${BASH_SOURCE[0]}
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_path")" && pwd -P)
if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf '%s\n' "Usage: spawn_random.sh [--variant density-only|density-velocity] [--frames N] [--seed N] [N]"
    exit 0
fi
sph_root=${SPH_ROOT:-${SPH_DIR:-${SPH_SIM_ROOT:-"$script_dir/../sph"}}}
if [[ ! -d "$sph_root" ]]; then
    printf 'SPH root does not exist: %s\n' "$sph_root" >&2
    exit 1
fi
sph_root=$(CDPATH= cd -- "$sph_root" && pwd -P)
data_root=${SPH_DATA_ROOT:-${DATA_ROOT:-"$script_dir/data"}}
mkdir -p -- "$data_root"
data_root=$(CDPATH= cd -- "$data_root" && pwd -P)
export SPH_DATA_ROOT="$data_root"

mode=${SPH_MODEL_VARIANT:-${SPH_VARIANT:-${MODEL_VARIANT:-${VARIANT:-density}}}}
frames=${FRAMES_PER_RUN:-${SPH_FRAMES_PER_RUN:-${FRAME_COUNT:-10000}}}
seed=${SPH_SCENARIO_SEED:-${SCENARIO_SEED:-${SPH_SEED:-${SEED:-0}}}}
extra_args=()
frames_seen=0
build_target_override=""
positional=()

usage() {
    printf '%s\n' "Usage: spawn_random.sh [--variant density-only|density-velocity] [--frames N] [--seed N] [N]" \
        "       spawn_random.sh [--variant MODE] [--frames N] [--seed N] [--fluid X Y W H] [--ghost X Y W H]"
}

require_value() {
    if (($# < 2)); then
        printf 'missing value for %s\n' "$1" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --variant|--model-variant|--model_variant|--mode|--simulation-mode|--simulation_mode)
            require_value "$@"
            mode=$2
            shift 2
            ;;
        --variant=*)
            mode=${1#*=}
            shift
            ;;
        --model-variant=*)
            mode=${1#*=}
            shift
            ;;
        --density-only)
            mode=density-only
            shift
            ;;
        --density-velocity)
            mode=density-velocity
            shift
            ;;
        --build-target|--target)
            require_value "$@"
            build_target_override=$2
            shift 2
            ;;
        --build-target=*)
            build_target_override=${1#*=}
            shift
            ;;
        --frames)
            require_value "$@"
            frames=$2
            frames_seen=1
            shift 2
            ;;
        --frames=*)
            frames=${1#*=}
            frames_seen=1
            shift
            ;;
        --seed|--scenario-seed)
            require_value "$@"
            seed=$2
            shift 2
            ;;
        --seed=*)
            seed=${1#*=}
            shift
            ;;
        --headless)
            shift
            ;;
        --fluid|-f|--ghost|-g)
            if (($# < 5)); then
                printf '%s requires four numeric values\n' "$1" >&2
                exit 2
            fi
            extra_args+=("$1" "$2" "$3" "$4" "$5")
            shift 5
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            while (($#)); do
                positional+=("$1")
                shift
            done
            ;;
        -*)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if ((${#positional[@]} > 1)); then
    printf 'only one legacy frame-count argument is accepted\n' >&2
    exit 2
fi
if ((${#positional[@]} == 1)); then
    if ((frames_seen)); then
        printf 'frame count was provided twice\n' >&2
        exit 2
    fi
    frames=${positional[0]}
    frames_seen=1
fi
if [[ ! $frames =~ ^[0-9]+$ || $frames =~ ^0+$ || ${#frames} -gt 9 ]]; then
    printf 'frame count must be a positive integer\n' >&2
    exit 2
fi
frames=$((10#$frames))
if [[ ! $seed =~ ^[0-9]+$ || ${#seed} -gt 18 ]]; then
    printf 'scenario seed must be a non-negative integer\n' >&2
    exit 2
fi
seed=$((10#$seed))

mode_key=$(printf '%s' "$mode" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
mode_key=${mode_key//-/_}
mode_key=${mode_key//+/_}
case "$mode_key" in
    density|density_only|densityonly|d)
        target=draw2-density-only
        metadata_variant=density-only
        canonical_variant=density
        ;;
    density_velocity|densityvelocity|dv|velocity|density+velocity)
        target=draw2-density-velocity
        metadata_variant=density-velocity
        canonical_variant=density_velocity
        ;;
    *)
        printf 'unsupported model variant: %s\n' "$mode" >&2
        exit 2
        ;;
esac
if [[ -n $build_target_override && $build_target_override != "$target" ]]; then
    printf 'requested build target %s does not match model variant target %s\n' "$build_target_override" "$target" >&2
    exit 2
fi
if [[ -n ${SPH_BUILD_TARGET:-} && $SPH_BUILD_TARGET != "$target" ]]; then
    printf 'requested build target %s does not match model variant target %s\n' "$SPH_BUILD_TARGET" "$target" >&2
    exit 2
fi
export SPH_BUILD_TARGET="$target"
export SPH_MODEL_VARIANT="$metadata_variant"
export SPH_MODEL_VARIANT_CANONICAL="$canonical_variant"
export SPH_VARIANT="$metadata_variant"
export SPH_SCENARIO_SEED="$seed"

binary_override=${SPH_BINARY:-${SPH_DRAW2:-}}
if [[ -n $binary_override ]]; then
    binary_dir=$(CDPATH= cd -- "$(dirname -- "$binary_override")" && pwd -P)
    binary="$binary_dir/$(basename -- "$binary_override")"
else
    binary="$sph_root/$target"
fi
if [[ $(basename -- "$binary") != "$target" ]]; then
    printf 'simulation binary %s does not match target %s\n' "$binary" "$target" >&2
    exit 2
fi
if [[ ! -x "$binary" ]]; then
    make_spec=${MAKE:-make}
    read -r -a make_command <<< "$make_spec"
    "${make_command[@]}" -C "$sph_root" "$target"
fi
if [[ ! -x "$binary" ]]; then
    printf 'simulation binary was not produced: %s\n' "$binary" >&2
    exit 1
fi

rng_state=$(((seed + 104729) & 0x7fffffff))
random_value=0
next_random() {
    rng_state=$(((rng_state * 1103515245 + 12345) & 0x7fffffff))
    random_value=$rng_state
}

next_random
scenario=$((random_value % 10))
next_random
offset=$((random_value % 40))
fluid_boxes=()
ghost_boxes=()

case "$scenario" in
    0)
        ghost_boxes=("90 350 140 30")
        fluid_boxes=("$((70 + offset)) 70 70 110")
        ;;
    1)
        ghost_boxes=("50 350 300 30")
        fluid_boxes=("$((55 + offset)) 60 45 120" "$((175 + offset)) 60 45 120" "$((285 + offset)) 60 45 120")
        ;;
    2)
        ghost_boxes=("70 220 55 55" "220 170 55 55" "300 260 55 55")
        fluid_boxes=("$((65 + offset)) 55 75 100")
        ;;
    3)
        ghost_boxes=("50 290 110 25" "150 230 110 25" "250 170 110 25")
        fluid_boxes=("$((60 + offset)) 45 80 100")
        ;;
    4)
        ghost_boxes=("45 310 75 22" "85 270 75 22" "125 230 75 22" "165 190 75 22" "205 150 75 22" "245 110 75 22")
        fluid_boxes=("$((55 + offset)) 40 65 85")
        ;;
    5)
        ghost_boxes=("80 315 240 28" "80 205 28 110" "292 205 28 110")
        fluid_boxes=("$((165 + offset)) 45 70 100")
        ;;
    6)
        ghost_boxes=("70 180 18 18" "125 155 18 18" "180 180 18 18" "235 155 18 18" "290 180 18 18" "105 235 18 18" "215 235 18 18")
        fluid_boxes=("$((160 + offset)) 40 80 90")
        ;;
    7)
        ghost_boxes=("45 180 125 22" "230 180 125 22" "45 205 22 105" "333 205 22 105")
        fluid_boxes=("$((165 + offset)) 40 70 100")
        ;;
    8)
        ghost_boxes=("80 140 20 130" "150 170 20 130" "220 140 20 130" "290 170 20 130")
        fluid_boxes=("$((55 + offset)) 40 70 100")
        ;;
    9)
        ghost_boxes=("45 350 310 30")
        fluid_boxes=("45 45 65 220" "290 45 65 220")
        ;;
esac

scenario_args=()
for box in "${fluid_boxes[@]}"; do
    read -r x y w h <<< "$box"
    scenario_args+=(--fluid "$x" "$y" "$w" "$h")
done
for box in "${ghost_boxes[@]}"; do
    read -r x y w h <<< "$box"
    scenario_args+=(--ghost "$x" "$y" "$w" "$h")
done
scenario_args+=("${extra_args[@]}")

printf 'Build target: %s\n' "$target"
printf 'Executing:'
printf ' %q' "$binary" "$frames" --headless "${scenario_args[@]}"
printf '\n'
exec "$binary" "$frames" --headless "${scenario_args[@]}"
