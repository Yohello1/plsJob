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
# module load cuda/12.4

export LD_LIBRARY_PATH=/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

source env/bin/activate
# --- Configuration ---
ITERATIONS=30         # Increased for 10-step Marathon
RUNS_PER_ITERATION=15 # Parallel simulations per cycle
MAX_SESSIONS=20       # Keep last N folders per run
FRAMES_PER_RUN=750    # Frames per simulation
MASS_LOSS_START_CYCLE=2
MASS_LOSS_WEIGHT=2.5
FLUID_LOSS_WEIGHT=35.0 # Increased to combat Zero-baseline drift
NOISE_STD=0.01        # Gaussian noise injected into inputs to improve drift stability
AR_STEPS=5            # Target sequence length for rollout training
AR_START_CYCLE=2      # Cycle to begin multi-step curriculum
AR_INCREMENT_INTERVAL=3 # Cycles to wait between increasing sequence length
SKIP_INITIAL=5        # Sparsely sample frames during initialization to speed up early cycles
USE_8BIT_ADAM=1       # Set to 1 to use BitsAndBytes 8-bit AdamW for VRAM savings

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
cd ../sph
# Only build if binary is missing or if explicitly requested via BUILD=1
make clean
make -j

cd ../neural_compression

echo "Starting Parallel Active Learning Loop for $RUN_NAME..."

for i in $(seq 1 $ITERATIONS); do
    echo "----------------------------------------"
    echo " Cycle $i of $ITERATIONS (Run: $RUN_NAME)"
    echo "----------------------------------------"

    # 0. DYNAMIC EPOCH CALCULATION
    # Increased epochs to allow better convergence with Spectral Normalization constraints
    if [ $i -lt 5 ]; then
        CURRENT_EPOCHS=5   # Fast initial adaptation
    elif [ $i -lt 15 ]; then
        CURRENT_EPOCHS=10  # Deepen learning as dataset grows
    else
        CURRENT_EPOCHS=20  # Fine-tuning on large cumulative data
    fi

    # 1. GENERATE DATA (Throttled Parallel execution)
    MAX_PARALLEL=3
    echo "Launching $RUNS_PER_ITERATION simulations ($MAX_PARALLEL at a time)..."
    for r in $(seq 1 $RUNS_PER_ITERATION); do
        # Run in background (&), redirect logs to the run-specific log folder
        ./spawn_random.sh $FRAMES_PER_RUN > "$LOG_DIR/sim_c${i}_r${r}.log" 2>&1 &
        
        # If we have reached the limit, wait for any one simulation to finish before starting the next
        if (( r >= MAX_PARALLEL )); then
            wait -n
        fi
    done

    # Wait for all remaining background simulation processes to finish
    wait
    echo "All simulations for cycle $i complete."

    # 2. STORAGE CLEANUP (Rolling Buffer - Run Specific)
    echo "Cleaning up old simulation data in $DATA_DIR (keeping top $MAX_SESSIONS)..."
    ls -dt "$DATA_DIR"/*/ | tail -n +$((MAX_SESSIONS + 1)) | xargs -r rm -rf

    # 3. TRAIN ON REMAINING DATA
    echo "Starting training session with $CURRENT_EPOCHS epochs..."
    ./env/bin/python compressor.py \
        --cycle $i \
        --epochs $CURRENT_EPOCHS \
        --data_dir "$DATA_DIR" \
        --output_dir "$ATTEMPTS_DIR" \
        --model_name "best_model.pth" \
        --mass_loss_weight $MASS_LOSS_WEIGHT \
        --mass_loss_start_cycle $MASS_LOSS_START_CYCLE \
        --fluid_weight $FLUID_LOSS_WEIGHT \
        --batch_size 0 \
        --effective_batch_size 8 \
        --skip_frames 5 \
        --n_steps $AR_STEPS \
        --ar_start_cycle $AR_START_CYCLE \
        --ar_increment_interval $AR_INCREMENT_INTERVAL \
        --noise_std $NOISE_STD \
        --skip_initial $SKIP_INITIAL \
        --use_8bit_adam \
        --bf16
     
    echo "Cycle $i complete."
done


echo "Active Training Loop Finished for $RUN_NAME."
