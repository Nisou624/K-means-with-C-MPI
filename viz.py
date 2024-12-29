import matplotlib.pyplot as plt
import numpy as np

# Read the output file
data_points = []
centroids = []

with open("output.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        if parts[0] == 'C':
            centroids.append((float(parts[1]), float(parts[2]), int(parts[3])))
        else:
            data_points.append((float(parts[0]), float(parts[1]), int(parts[2])))

# Create a scatter plot
plt.figure(figsize=(13, 6))

# Define a color map
colors = plt.cm.get_cmap('tab10', len(centroids))

# Plot data points
for point in data_points:
    cluster_id = point[2]
    plt.scatter(point[0], point[1], c=[colors(cluster_id)], marker='o')

# Plot centroids
for centroid in centroids:
    plt.scatter(centroid[0], centroid[1], c=[colors(centroid[2])], marker='x', s=100, linewidths=2)

# Create legend handles and labels
handles = []
labels = []
for i in range(len(centroids)):
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10))
    labels.append(f'Cluster {i}')

plt.title("K-means Clustering Results")
plt.xlabel("X")
plt.ylabel("Y")

# Place the legend outside the plot area
plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left')

plt.show()
