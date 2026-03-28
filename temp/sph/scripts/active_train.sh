#!/bin/bash
# High-accuracy Active Learning Loop for SPH Autoencoder (Storage-Optimized)

# Configuration
ITERATIONS=20         # Total cycles of (Simulate + Train)
RUNS_PER_ITERATION=5  # How many simulations to start per cycle
MAX_SESSIONS=5        # Keep only the last N simulation folders per run
EPOCHS_PER_CLEAN=5   # Number of training epochs per cycle
FRAMES_PER_RUN=500    # Number of frames per simulation run

# --- Unique Run Setup ---
# Use first argument as RUN_NAME if provided, otherwise generate a unique one
RUN_NAME=${1:-run_$(date +%Y%m%d_%H%M%S)}
DATA_DIR="data/$RUN_NAME"
LOG_DIR="logs/$RUN_NAME"
ATTEMPTS_DIR="attempts/$RUN_NAME"

echo "Setting up unique run: $RUN_NAME"
mkdir -p "$DATA_DIR" "$LOG_DIR" "$ATTEMPTS_DIR"

# Export for the simulation binary (src/logging.cpp)
export SPH_DATA_ROOT="$DATA_DIR"

echo "Checking fluid sim build...";
cd ..
if [ ! -f ./draw2 ] || [ "$BUILD" == "1" ]; then
    echo "Building (this might take a moment)..."
    make -j
fi
cd scripts

echo "Starting Active Learning Loop for $RUN_NAME..."

for i in $(seq 1 $ITERATIONS); do
    echo "========================================"
    echo "   Cycle $i of $ITERATIONS (Run: $RUN_NAME)"
    echo "========================================"
    
    # 1. GENERATE DATA
    for r in $(seq 1 $RUNS_PER_ITERATION); do
        echo "Running simulation $r of $RUNS_PER_ITERATION for $FRAMES_PER_RUN frames..."
        ./spawn_random.sh $FRAMES_PER_RUN > "$LOG_DIR/sim_c${i}_r${r}.log" 2>&1
    done

    # 2. STORAGE CLEANUP (Rolling Buffer - Run Specific)
    echo "Cleaning up old simulation data in $DATA_DIR (keeping top $MAX_SESSIONS)..."
    ls -dt "$DATA_DIR"/*/ | tail -n +$((MAX_SESSIONS + 1)) | xargs -r rm -rf

    # 3. TRAIN ON REMAINING DATA
    echo "Starting training session on all remaining data in $DATA_DIR..."
    python compressor.py \
        --epochs $EPOCHS_PER_CLEAN \
        --data_dir "$DATA_DIR" \
        --output_dir "$ATTEMPTS_DIR" \
        --model_name "best_model.pth"
    
    echo "Cycle $i complete. Best model checkpoint updated in $ATTEMPTS_DIR."
done

echo "Active Training Loop Finished for $RUN_NAME."
