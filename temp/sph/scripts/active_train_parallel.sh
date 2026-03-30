#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --output=logs/cuda_test_%j.log
#SBATCH --partition=gpu-gen
#SBATCH --nodelist=gpu-pt1-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00

# --- Environment Setup ---
# Load the verified CUDA module
module load cuda/12.4

source env/bin/activate
# --- Configuration ---
ITERATIONS=20         # Total cycles
RUNS_PER_ITERATION=10 # Parallel simulations per cycle
MAX_SESSIONS=30       # Keep last N folders per run
EPOCHS_PER_CLEAN=7    # Training epochs per cycle
FRAMES_PER_RUN=200    # Frames per simulation
MASS_LOSS_START_CYCLE=5
MASS_LOSS_WEIGHT=2.0

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
# Only build if binary is missing or if explicitly requested via BUILD=1
make clean
make -j

cd scripts

echo "Starting Parallel Active Learning Loop for $RUN_NAME..."

for i in $(seq 1 $ITERATIONS); do
    echo "----------------------------------------"
    echo " Cycle $i of $ITERATIONS (Run: $RUN_NAME)"
    echo "----------------------------------------"

    # 1. GENERATE DATA (Parallel execution)
    echo "Launching $RUNS_PER_ITERATION simulations in parallel..."
    for r in $(seq 1 $RUNS_PER_ITERATION); do
        # Run in background (&), redirect logs to the run-specific log folder
        ./spawn_random.sh $FRAMES_PER_RUN > "$LOG_DIR/sim_c${i}_r${r}.log" 2>&1 &
    done

    # Wait for all background simulation processes to finish
    wait
    echo "All simulations for cycle $i complete."

    # 2. STORAGE CLEANUP (Rolling Buffer - Run Specific)
    echo "Cleaning up old simulation data in $DATA_DIR (keeping top $MAX_SESSIONS)..."
    ls -1dt "$DATA_DIR"/*/ | tail -n +$((MAX_SESSIONS + 1)) | xargs -r rm -rf

    # 3. TRAIN ON REMAINING DATA
    echo "Starting training session..."
    ./env/bin/python compressor.py \
        --cycle $i \
        --epochs $EPOCHS_PER_CLEAN \
        --data_dir "$DATA_DIR" \
        --output_dir "$ATTEMPTS_DIR" \
        --model_name "best_model.pth" \
        --mass_loss_weight $MASS_LOSS_WEIGHT \
        --mass_loss_start_cycle $MASS_LOSS_START_CYCLE \
        --batch_size 0 \
        --effective_batch_size 8 \
        --bf16
    
    echo "Cycle $i complete."
done

echo "Active Training Loop Finished for $RUN_NAME."
