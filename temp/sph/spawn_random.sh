#!/bin/bash

# SPH Simulation Random Spawner
# Values based on settings.hpp and user modifications:
# Working area (unpadded) is roughly 40 to 760/770

MIN_X=40
MAX_X=760
MIN_Y=40
MAX_Y=770

# Choose a random scenario
# 0: Single Fluid + Static Floor
# 1: Multiple Fluid Drops
# 2: Floating Obstacle Course (Ghost boxes in space)
# 3: Complex Multi-level Floor
# 4: Stair Steps
# 5: Central Bowl
SCENARIO=$((RANDOM % 6))

FLUID_ARGS=""
GHOST_ARGS=""

case $SCENARIO in
  0)
    echo "Scenario: Single Fluid + Static Floor"
    FW=$((RANDOM % 200 + 150))
    FH=$((RANDOM % 200 + 150))
    FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
    FY=$((RANDOM % (400 - FH - MIN_Y) + MIN_Y))
    FLUID_ARGS="--fluid $FX $FY $FW $FH"
    
    # Generic bottom floor
    GW=$((RANDOM % 300 + 200))
    GH=40
    GX=$((RANDOM % (MAX_X - GW - MIN_X) + MIN_X))
    GY=$((MAX_Y - GH - 10))
    GHOST_ARGS="--ghost $GX $GY $GW $GH"
    ;;
    
  1)
    echo "Scenario: Multiple Fluid Drops"
    NUM_FLUIDS=$((RANDOM % 4 + 3)) # 3 to 7 drops
    for i in $(seq 1 $NUM_FLUIDS); do
      FW=$((RANDOM % 80 + 40))
      FH=$((RANDOM % 80 + 40))
      FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
      FY=$((RANDOM % (500 - FH - MIN_Y) + MIN_Y))
      FLUID_ARGS="$FLUID_ARGS --fluid $FX $FY $FW $FH"
    done
    # Solid bottom floor to catch them
    GHOST_ARGS="--ghost $MIN_X $((MAX_Y - 20)) $((MAX_X - MIN_X)) 20"
    ;;
    
  2)
    echo "Scenario: Floating Obstacle Course"
    # Large fluid reservoir at the top
    FW=400; FH=150
    FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
    FY=$MIN_Y
    FLUID_ARGS="--fluid $FX $FY $FW $FH"
    
    # 5-8 random floating ghost blocks ("in space")
    NUM_GHOSTS=$((RANDOM % 4 + 5))
    for i in $(seq 1 $NUM_GHOSTS); do
      GW=$((RANDOM % 100 + 40))
      GH=$((RANDOM % 60 + 30))
      GX=$((RANDOM % (MAX_X - GW - MIN_X) + MIN_X))
      # Spread them vertically between top reservoir and bottom
      GY=$((RANDOM % (MAX_Y - GH - 200 - MIN_Y) + MIN_Y + 150))
      GHOST_ARGS="$GHOST_ARGS --ghost $GX $GY $GW $GH"
    done
    ;;
    
  3)
    echo "Scenario: Complex Multi-level Floor"
    # Fluid box
    FW=300; FH=200
    FX=$((RANDOM % (MAX_X - FW - MIN_X) + MIN_X))
    FY=$MIN_Y
    FLUID_ARGS="--fluid $FX $FY $FW $FH"
    
    # Multiple overlapping or tiered ghost boxes at the bottom
    NUM_GHOSTS=$((RANDOM % 5 + 4))
    for i in $(seq 1 $NUM_GHOSTS); do
      GW=$((RANDOM % 150 + 50))
      GH=$((RANDOM % 100 + 20))
      GX=$((RANDOM % (MAX_X - GW - MIN_X) + MIN_X))
      GY=$((MAX_Y - GH - 10))
      GHOST_ARGS="$GHOST_ARGS --ghost $GX $GY $GW $GH"
    done
    ;;

  4)
    echo "Scenario: Stair Steps"
    # One reservoir at the top left/right
    FW=200; FH=150
    # Steps will go top-down
    DIR=$((RANDOM % 2)) # 0: left to right, 1: right to left
    if [ $DIR -eq 0 ]; then
      FX=60; FX_START=40
    else
      FX=500; FX_START=700
    fi
    FY=50
    FLUID_ARGS="--fluid $FX $FY $FW $FH"
    
    # Steps
    NUM_STEPS=$((RANDOM % 4 + 6)) # 6-10 steps
    STEP_W=120
    STEP_H=20
    for i in $(seq 0 $((NUM_STEPS-1))); do
      if [ $DIR -eq 0 ]; then
        GX=$((FX_START + i * 70))
      else
        GX=$((FX_START - i * 70 - STEP_W))
      fi
      GY=$((150 + i * 60))
      GHOST_ARGS="$GHOST_ARGS --ghost $GX $GY $STEP_W $STEP_H"
    done
    ;;

  5)
    echo "Scenario: Central Bowl"
    # Reservoir at the top center
    FW=250; FH=200
    FX=$((400 - FW/2))
    FY=60
    FLUID_ARGS="--fluid $FX $FY $FW $FH"
    
    # The Bowl
    BW=400; BH=150
    BX=$((400 - BW/2))
    BY=$((MAX_Y - BH - 50))
    # Bottom
    GHOST_ARGS="--ghost $BX $BY $BW 30"
    # Left wall
    GHOST_ARGS="$GHOST_ARGS --ghost $BX $((BY - BH)) 30 $BH"
    # Right wall
    GHOST_ARGS="$GHOST_ARGS --ghost $((BX + BW - 30)) $((BY - BH)) 30 $BH"
    ;;
esac

echo "Executing: ./draw2 $FLUID_ARGS $GHOST_ARGS"
./draw2 $FLUID_ARGS $GHOST_ARGS
