import matplotlib.pyplot as plt
import re
import os

def generate_frame_graph(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Read the file content
    with open(file_path, 'r') as file:
        data = file.read()

    # Regex to find "Frame X Time: Ys"
    # Group 1: Frame Number, Group 2: Time in seconds
    pattern = r"Frame\s+(\d+)\s+Time:\s+([\d.]+)s"
    matches = re.findall(pattern, data)
    
    if not matches:
        print("No valid frame data found in the file.")
        return

    frames = [int(m[0]) for m in matches]
    times = [float(m[1]) for m in matches]

    # Calculate statistics
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(frames, times, label='Raw Frame Time', color='#3498db', alpha=0.4, linewidth=1)
    
    # Optional: Add a rolling average to see trends clearly (window of 10 frames)
    if len(times) > 10:
        rolling_avg = [sum(times[i:i+10])/10 for i in range(len(times)-10)]
        plt.plot(frames[10:], rolling_avg, label='Trend (10-frame Avg)', color='#e74c3c', linewidth=2)

    # Formatting the Graph
    plt.title(f'Performance Analysis: {file_path}', fontsize=16, pad=20)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Render Time (seconds)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Horizontal line for average
    plt.axhline(y=avg_time, color='green', linestyle='--', label=f'Overall Avg: {avg_time:.3f}s')

    # Adding a text box with stats
    stats_text = f"Total Frames: {len(frames)}\nMax: {max_time:.3f}s\nMin: {min_time:.3f}s"
    plt.gca().annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                       bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the result as an image or just show it
    plt.savefig('frame_time_analysis.png', dpi=300)
    print("Graph saved as 'frame_time_analysis.png'")
    plt.show()

# Execution
if __name__ == "__main__":
    # Ensure data.txt is in the same folder as this script
    generate_frame_graph('data.txt')
