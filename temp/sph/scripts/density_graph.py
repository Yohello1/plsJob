import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Constants based on SPH settings (as found in settings.hpp and compressor.py)
BUFFER_WIDTH = 400
BUFFER_HEIGHT = 400
NUM_FIELDS = 4
FIELD_SIZE = BUFFER_WIDTH * BUFFER_HEIGHT
FRAME_SIZE_FLOATS = NUM_FIELDS * FIELD_SIZE
FRAME_SIZE_BYTES = FRAME_SIZE_FLOATS * 4

def main():
    if len(sys.argv) < 2:
        print("Usage: python density_graph.py <path_to_sim_data.bin>")
        print("Note: If you have a directory, point to the sim_data.bin inside it.")
        sys.exit(1)

    bin_file = sys.argv[1]
    
    # If a directory is provided, look for sim_data.bin inside
    if os.path.isdir(bin_file):
        bin_file = os.path.join(bin_file, "sim_data.bin")

    if not os.path.exists(bin_file):
        print(f"Error: File not found: {bin_file}")
        sys.exit(1)

    file_size = os.path.getsize(bin_file)
    num_frames = file_size // FRAME_SIZE_BYTES
    
    if num_frames == 0:
        print("No complete frames found in file.")
        if file_size > 0:
            print(f"File size: {file_size} bytes. Required for one frame: {FRAME_SIZE_BYTES} bytes.")
        sys.exit(1)

    print(f"Found {num_frames} frames in {bin_file}")
    print("-" * 30)
    print(f"{'Frame':>8} | {'Sum of Densities':>18}")
    print("-" * 30)
    
    densities = []
    
    try:
        with open(bin_file, "rb") as f:
            for i in range(num_frames):
                # Seek to the start of the frame's density field
                f.seek(i * FRAME_SIZE_BYTES)
                
                # Read the density field (first field of the frame)
                field_data = np.fromfile(f, dtype=np.float32, count=FIELD_SIZE)
                
                if field_data.size != FIELD_SIZE:
                    print(f"\nWarning: Incomplete frame {i}")
                    break
                    
                frame_sum = np.sum(field_data)
                densities.append(frame_sum)
                
                # Output to terminal
                print(f"{i:8d} | {frame_sum:18.6f}")
                
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    if not densities:
        print("No data collected.")
        sys.exit(1)

    # Graphing
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(densities)), densities, marker='.', linestyle='-', color='#007acc', alpha=0.7)
    
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Sum of Densities (Average Density per Cell Sum)', fontsize=12)
    plt.title(f'Density Evolution: {os.path.basename(os.path.dirname(bin_file))}', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Styling
    plt.tight_layout()
    
    output_plot = "density_evolution.png"
    plt.savefig(output_plot, dpi=300)
    print("-" * 30)
    print(f"\nGraph saved to: {os.path.abspath(output_plot)}")
    
    # Try to show the plot (might not work in all environments, but good to have)
    try:
        plt.show()
    except Exception:
        print("Could not display plot window (likely headless environment). Image has been saved.")

if __name__ == "__main__":
    main()
