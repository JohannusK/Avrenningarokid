import os
from utils import find_steepest_descent, loadMap, plotFigure
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import alphashape

# Bounds in the format (west, south, east, north)
bounds = (-6.351157171442581, 62.30967342287517, -6.2411181682078976, 62.360487093395776)
file_path = 'FO_DSM_2017_FOTM_10M.tif'

image_array, img_pil = loadMap(file_path, bounds)
height, width = image_array.shape

grid_size = 200

paths = []
grid_x = np.linspace(0, image_array.shape[0] - 1, grid_size)
grid_y = np.linspace(0, image_array.shape[1] - 1, grid_size)

start_positions = [(int(x), int(y)) for x in grid_x for y in grid_y]

for x in grid_x:
    for y in grid_y:
        start_pos = (int(x), int(y))
        path = find_steepest_descent(image_array, start_pos, max_radius=40)
        if len(path) > 1:
            paths.append(path)

# Calculate the length of each path
path_lengths = [len(path) for path in paths]

# Replicate end points based on path lengths
weighted_end_points = []
original_end_points = []
for path, length in zip(paths, path_lengths):
    weighted_end_points.extend([path[-1]] * length)
    original_end_points.append(path[-1])

weighted_end_points = np.array(weighted_end_points)
original_end_points = np.array(original_end_points)

# Create directory for saving figures
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)


# Function to calculate coverage percentage and create plots for a given number of clusters
def calculate_coverage_percentage_and_plot(n_clusters, weighted_end_points, original_end_points, paths, height,
                                           range_in_meters=100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weighted_end_points)
    kmeans_labels = kmeans.predict(original_end_points)
    unique_labels = set(kmeans_labels)

    representative_points = []
    cluster_start_points = 0
    total_start_points = len(paths)

    # Plot initial image and contours
    plt.figure(figsize=(10, 6))
    plt.imshow(img_pil, aspect='equal')
    contour_levels = [0, 100, 150, 200, 250, 300]
    plt.contour(image_array, levels=contour_levels, colors='black', linewidths=1, origin='image')
    plt.gca().invert_yaxis()  # Adjust axis if necessary depending on data orientation
    plt.axis('off')  # Turn off the axis

    for k, col in zip(unique_labels, plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))):
        class_member_mask = (kmeans_labels == k)
        cluster_end_points = original_end_points[class_member_mask]
        centroid = cluster_end_points.mean(axis=0)

        # Find the nearest endpoint to the centroid
        distances = np.linalg.norm(cluster_end_points - centroid, axis=1)
        nearest_index = np.argmin(distances)
        representative_point = cluster_end_points[nearest_index]
        representative_points.append(representative_point)

        # Plot the clustered end points with colors
        xy = cluster_end_points
        plt.plot(xy[:, 1], xy[:, 0] * -1 + height, 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=4, alpha=0.6)

    # Plot paths and calculate the number of paths within the range
    path_assigned_to_cluster = np.zeros(len(paths), dtype=bool)
    for k, col in zip(unique_labels, plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))):
        representative_point = representative_points[k]
        cluster_start_points_k = 0
        for path_index, path in enumerate(paths):
            end_point = path[-1]
            distance = np.linalg.norm(end_point - representative_point)
            x_coords = [coord[1] for coord in path]
            y_coords = [coord[0] * -1 + height for coord in path]
            if distance <= range_in_meters / 10:  # 100 meters (10 in 10x10 meter grid)
                plt.plot(x_coords, y_coords, color=col, alpha=0.6)
                path_assigned_to_cluster[path_index] = True
                cluster_start_points_k += 1
            else:
                plt.plot(x_coords, y_coords, 'black', alpha=0.01)

        # Calculate alpha shape for the first points
        cluster_paths = [path for path, label in zip(paths, kmeans_labels) if label == k]
        first_points = np.array([path[0] for path in cluster_paths])
        first_points[:, 0] = height - first_points[:, 0]  # Adjust y-coordinates
        alpha_shape = alphashape.alphashape(first_points, 0.1)

        # Plot the alpha shape with the same color as the cluster markers
        if alpha_shape.geom_type == 'Polygon':
            x, y = alpha_shape.exterior.xy
            plt.plot(y, x, color=col, linestyle='-', linewidth=2)
        elif alpha_shape.geom_type == 'MultiPolygon':
            for polygon in alpha_shape:
                x, y = polygon.exterior.xy
                plt.plot(y, x, color=col, linestyle='-', linewidth=2)
        else:
            print("Alpha shape resulted in an empty geometry or invalid geometry:", alpha_shape)

    cluster_start_points = np.sum(path_assigned_to_cluster)
    coverage_percentage = (cluster_start_points / total_start_points) * 100

    # Plot the representative endpoints
    for representative_point in representative_points:
        plt.plot(representative_point[1], representative_point[0] * -1 + height, 'o', markerfacecolor='red',
                 markeredgecolor='k', markersize=12, alpha=0.9)

    plt.title(f'Paths with K-Means Clustering, Alpha Shapes, and Representative End Points\n'
              f'Clusters: {n_clusters}, Coverage: {coverage_percentage:.2f}%')

    # Save the figure
    plt.savefig(f"{figures_dir}/coverage_clusters_{n_clusters:03d}.png")
    plt.close()

    return coverage_percentage


# Calculate and store coverage percentages for different number of clusters
n_clusters_range = range(1, 101)
coverage_percentages = []

for n_clusters in n_clusters_range:
    coverage_percentage = calculate_coverage_percentage_and_plot(n_clusters, weighted_end_points, original_end_points,
                                                                 paths, height, range_in_meters=100)
    coverage_percentages.append(coverage_percentage)
    print(f"Number of clusters: {n_clusters}, Coverage percentage: {coverage_percentage:.2f}%")

# Final plot of the number of clusters versus the coverage percentage
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, coverage_percentages, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Coverage Percentage (%)')
plt.title('Coverage Percentage vs. Number of Clusters')
plt.grid(True)
plt.savefig(f"{figures_dir}/coverage_clusters_final.png")
plt.show()
