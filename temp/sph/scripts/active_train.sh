#!/bin/bash
# High-accuracy Active Learning Loop for SPH Autoencoder (Storage-Optimized)

# Configuration
ITERATIONS=20         # Total cycles of (Simulate + Train)
RUNS_PER_ITERATION=2  # How many simulations to start per cycle
MAX_SESSIONS=5        # Keep only the last 5 simulation folders (to save space)
EPOCHS_PER_CLEAN=10   # Number of training epochs per cycle
FRAMES_PER_RUN=500    # Number of frames per simulation run

# Ensure directories exist
mkdir -p data

echo "Starting Active Learning Loop..."
echo "Storage Limit: Last $MAX_SESSIONS simulation sessions will be preserved."

for i in $(seq 1 $ITERATIONS); do
    echo "========================================"
    echo "   Cycle $i of $ITERATIONS"
    echo "========================================"
    
    # 1. GENERATE DATA
    for r in $(seq 1 $RUNS_PER_ITERATION); do
        echo "Running simulation $r of $RUNS_PER_ITERATION for $FRAMES_PER_RUN frames..."
        ./spawn_random.sh $FRAMES_PER_RUN > /dev/null 2>&1
    done

    # 2. STORAGE CLEANUP (Rolling Buffer)
    # This finds all subdirectories in data/ (excluding data/ itself and data/frames)
    # Sorts them by modification time (newest first)
    # Picks everything after the 5th and deletes them
    echo "Cleaning up old simulation data (keeping top $MAX_SESSIONS)..."
    ls -dt data/*/ | tail -n +$((MAX_SESSIONS + 1)) | xargs -r rm -rf

    # 3. TRAIN ON REMAINING DATA (all current sessions)
    echo "Starting training session on all remaining data..."
    python compressor.py --epochs $EPOCHS_PER_CLEAN
    
    echo "Cycle $i complete. Best model checkpoint updated."
done

echo "Active Training Loop Finished."
