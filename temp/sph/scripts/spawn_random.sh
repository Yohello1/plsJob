#!/bin/bash
# SPH Simulation Random Spawner (400x400 Resolution Optimized)
# Increased diversity for better autoencoder generalization

# Seed RANDOM with nanoseconds + PID
RANDOM=$(( (10#$(date +%N | cut -b4-9) + $$) % 32768 ))

# Resolution Boundaries (400x400)
MIN_X=50
MAX_X=350
MIN_Y=50
MAX_Y=370

# Scenario Choice (Increased to 10 types)
SCENARIO=$((RANDOM % 10))
FLUID_BOXES=()
GHOST_BOXES=()

# Helper to check if two rects intersect
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
  6)
    # Plinko / Peg Board (Forces fluid to split)
    for r in {0..3}; do
        for c in {0..5}; do
            OFFSET=$(( (r % 2) * 25 ))
            GHOST_BOXES+=("$((70 + c * 50 + OFFSET)) $((150 + r * 50)) 15 15")
        done
    done
    ;;
  7)
    # Hourglass / Funnel (High pressure jet)
    GHOST_BOXES+=("50 200 120 20")  # Left slope
    GHOST_BOXES+=("230 200 120 20") # Right slope
    GHOST_BOXES+=("50 220 20 100")  # Left wall
    GHOST_BOXES+=("330 220 20 100") # Right wall
    ;;
  8)
    # Random Pillars
    for i in {1..8}; do
        GHOST_BOXES+=("$((RANDOM % 250 + 70)) $((RANDOM % 200 + 100)) 15 $((RANDOM % 60 + 20))")
    done
    ;;
  9)
    # Two Dam Breaks (Colliding fluid)
    GHOST_BOXES+=("$MIN_X 360 $((MAX_X - MIN_X)) 20") # Floor
    FLUID_BOXES+=("50 50 60 200") # Left wall of water
    FLUID_BOXES+=("290 50 60 200") # Right wall of water
    ;;
esac

# Try placing fluid boxes
# Randomly decide how many drops to spawn (between 1 and 3)
NUM_DROP_TARGETS=$(( (RANDOM % 3) + 1 ))
if [ $SCENARIO -eq 1 ]; then NUM_DROP_TARGETS=5; fi
if [ $SCENARIO -eq 9 ]; then NUM_DROP_TARGETS=0; fi # Already handled

for (( i=1; i<=NUM_DROP_TARGETS; i++ )); do
    for attempt in {1..20}; do
        # 30% chance of spawning a "cluster" (blobby irregular shape)
        if [ $((RANDOM % 10)) -lt 3 ]; then
            CX=$((RANDOM % (MAX_X - 100 - MIN_X) + MIN_X))
            CY=$((RANDOM % (200 - MIN_Y) + MIN_Y))
            
            COLLISION=false
            TEMP_BLOBS=()
            for blob in {1..3}; do
                FW=$((RANDOM % 40 + 30))
                FH=$((RANDOM % 40 + 30))
                FX=$((CX + RANDOM % 30))
                FY=$((CY + RANDOM % 30))
                
                # Check each blob against all ghost boxes
                for gbox in "${GHOST_BOXES[@]}"; do
                    read gx gy gw gh <<< "$gbox"
                    if check_intersect $FX $FY $FW $FH $gx $gy $gw $gh; then
                        COLLISION=true; break 2
                    fi
                done
                TEMP_BLOBS+=("$FX $FY $FW $FH")
            done
            
            if [ "$COLLISION" = false ]; then
                for b in "${TEMP_BLOBS[@]}"; do FLUID_BOXES+=("$b"); done
                break
            fi
        else
            # Normal rectangle drop
            FW=$((RANDOM % 80 + 40))
            FH=$((RANDOM % 80 + 40))
            FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
            FY=$((RANDOM % (250 - FH - MIN_Y) + MIN_Y))
            
            COLLISION=false
            for gbox in "${GHOST_BOXES[@]}"; do
                read gx gy gw gh <<< "$gbox"
                if check_intersect $FX $FY $FW $FH $gx $gy $gw $gh; then
                    COLLISION=true; break
                fi
            done
            if [ "$COLLISION" = false ]; then
                FLUID_BOXES+=("$FX $FY $FW $FH")
                break
            fi
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
../draw2 $FRAME_COUNT $FLUID_ARGS $GHOST_ARGS
