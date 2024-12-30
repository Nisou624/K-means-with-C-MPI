import matplotlib.pyplot as plt
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python3 generate_graph.py <results_file>")
    sys.exit(1)

# Read the results file
results_file = sys.argv[1]
sequential_times = []
parallel_times = []
max_iterations_values = []

with open(results_file, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        if parts[0] == "Sequential":
            max_iterations_values.append(int(parts[1]))
            sequential_times.append(float(parts[2]))
        elif parts[0] == "Parallel":
            parallel_times.append(float(parts[2]))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_iterations_values, sequential_times, marker='o', label='Sequential')
plt.plot(max_iterations_values, parallel_times, marker='x', label='Parallel')
plt.xlabel('MAX_ITERATIONS')
plt.ylabel('Average Execution Time (s)')
plt.title('K-means Performance Comparison')
plt.legend()
plt.grid(True)
plt.savefig('results/performance_comparison.png')
plt.show()
