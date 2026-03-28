#!/bin/bash
# SPH Simulation Random Spawner (400x400 Resolution Optimized)
# Prevents fluid from intersecting with ghost boxes

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    RANDOM=$SLURM_ARRAY_TASK_ID
fi

# Resolution Boundaries (400x400)
MIN_X=50
MAX_X=350
MIN_Y=50
MAX_Y=370

# Scenario Choice
SCENARIO=$((RANDOM % 6))
FLUID_BOXES=()
GHOST_BOXES=()

# Helper to check if two rects intersect
# Arguments: x1 y1 w1 h1 x2 y2 w2 h2
check_intersect() {
    local x1=$1; local y1=$2; local w1=$3; local h1=$4
    local x2=$5; local y2=$6; local w2=$7; local h2=$8

    if (( x1 < x2 + w2 )) && (( x1 + w1 > x2 )) && \
       (( y1 < y2 + h2 )) && (( y1 + h1 > y2 )); then
        return 0 # True (Collision)
    fi
    return 1 # False
}

# Scenario Logic (Placement of Ghosts First)
case $SCENARIO in
  0)
    # Single Fluid + Floor
    GHOST_BOXES+=("$((RANDOM % 150 + 100)) 340 $((RANDOM % 150 + 100)) 40")
    ;;
  1)
    # Multiple Drop Scenario
    GHOST_BOXES+=("$MIN_X 360 $((MAX_X - MIN_X)) 20")
    ;;
  2)
    # Floating Obstacle Course
    for i in {1..5}; do
        GHOST_BOXES+=("$((RANDOM % 200 + 50)) $((RANDOM % 200 + 100)) $((RANDOM % 60 + 30)) $((RANDOM % 40 + 20))")
    done
    ;;
  3)
    # Complex Multi-level
    for i in {1..4}; do
        GHOST_BOXES+=("$((RANDOM % 150 + 50)) $((MAX_Y - 50)) $((RANDOM % 100 + 50)) $((RANDOM % 40 + 10))")
    done
    ;;
  4)
    # Stair Steps
    for i in {0..6}; do
        GHOST_BOXES+=("$((60 + i * 40)) $((150 + i * 30)) 80 20")
    done
    ;;
  5)
    # Central Bowl
    GHOST_BOXES+=("100 300 200 30") # Bottom
    GHOST_BOXES+=("100 200 30 100") # Left
    GHOST_BOXES+=("270 200 30 100") # Right
    ;;
esac

# Try placing fluid boxes (Max 10 attempts to find a non-intersecting spot)
NUM_DROP_TARGETS=$((SCENARIO == 1 ? 5 : 1))
for (( i=1; i<=NUM_DROP_TARGETS; i++ )); do
    PLACED=false
    for attempt in {1..20}; do
        FW=$((RANDOM % 80 + 40))
        FH=$((RANDOM % 80 + 40))
        FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
        FY=$((RANDOM % (250 - FH - MIN_Y) + MIN_Y))

        COLLISION=false
        for gbox in "${GHOST_BOXES[@]}"; do
            read gx gy gw gh <<< "$gbox"
            if check_intersect $FX $FY $FW $FH $gx $gy $gw $gh; then
                COLLISION=true
                break
            fi
        done

        if [ "$COLLISION" = false ]; then
            FLUID_BOXES+=("$FX $FY $FW $FH")
            PLACED=true
            break
        fi
    done
done

# Assemble Arguments
FLUID_ARGS=""
for fbox in "${FLUID_BOXES[@]}"; do
    FLUID_ARGS="$FLUID_ARGS --fluid $fbox"
done

GHOST_ARGS=""
for gbox in "${GHOST_BOXES[@]}"; do
    GHOST_ARGS="$GHOST_ARGS --ghost $gbox"
done

# Run Simulation
FRAME_COUNT=${1:-10000}
echo "Executing: ./draw2 $FRAME_COUNT --headless $FLUID_ARGS $GHOST_ARGS"
../draw2 $FRAME_COUNT --headless $FLUID_ARGS $GHOST_ARGS

