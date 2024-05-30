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

plotFigure(img_pil, image_array)

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
            x_coords = [coord[1] for coord in path]
            y_coords = [coord[0] * -1 + height for coord in path]
            paths.append(path)

# Calculate the length of each path
path_lengths = [len(path) for path in paths]

# Replicate end points based on path lengths
weighted_end_points = []
for path, length in zip(paths, path_lengths):
    weighted_end_points.extend([path[-1]] * length)

weighted_end_points = np.array(weighted_end_points)

# K-Means Clustering on weighted data
kmeans = KMeans(n_clusters=4, random_state=0).fit(weighted_end_points)
kmeans_labels = kmeans.predict(np.array([path[-1] for path in paths]))

# Extract original end points
end_points = np.array([path[-1] for path in paths])

# Plotting the combined results on the original map
unique_labels = set(kmeans_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# Plot the clustered end points with colors
for k, col in zip(unique_labels, colors):
    class_member_mask = (kmeans_labels == k)
    xy = end_points[class_member_mask]
    plt.plot(xy[:, 1], xy[:, 0] * -1 + height, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4, alpha=0.6)

# Loop through each cluster and draw alpha shapes for the first points of paths
alpha = 0.0000000001  # Adjust alpha value if needed
for k, col in zip(unique_labels, colors):
    cluster_paths = [path for path, label in zip(paths, kmeans_labels) if label == k]
    first_points = np.array([path[0] for path in cluster_paths])
    first_points[:, 0] = height - first_points[:, 0]  # Adjust y-coordinates

    # Compute the alpha shape of the first points
    alpha_shape = alphashape.alphashape(first_points, alpha)

    # Plot the alpha shape with the same color as the cluster markers
    if alpha_shape.geom_type == 'Polygon':
        x, y = alpha_shape.exterior.xy
        plt.plot(y, x, color=tuple(col), linestyle='-', linewidth=2)
    elif alpha_shape.geom_type == 'MultiPolygon':
        for polygon in alpha_shape:
            x, y = polygon.exterior.xy
            plt.plot(y, x, color=tuple(col), linestyle='-', linewidth=2)
    else:
        print("Alpha shape resulted in an empty geometry or invalid geometry:", alpha_shape)

# Choose a representative end point for each cluster by finding the nearest to the centroid
representative_points = []
for k, col in zip(unique_labels, colors):
    class_member_mask = (kmeans_labels == k)
    cluster_end_points = end_points[class_member_mask]
    centroid = cluster_end_points.mean(axis=0)

    # Find the nearest endpoint to the centroid
    distances = np.linalg.norm(cluster_end_points - centroid, axis=1)
    nearest_index = np.argmin(distances)
    representative_point = cluster_end_points[nearest_index]
    representative_points.append(representative_point)

    # Plot the representative endpoint with a larger marker
    plt.plot(representative_point[1], representative_point[0] * -1 + height, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=12, alpha=0.9)

# Plot all paths, coloring those within 50 meters of the representative point
total_start_points = 0
cluster_start_points = 0

for k, col in zip(unique_labels, colors):
    representative_point = representative_points[k]
    cluster_start_points_k = 0
    for path in paths:
        end_point = path[-1]
        distance = np.linalg.norm(end_point - representative_point)
        x_coords = [coord[1] for coord in path]
        y_coords = [coord[0] * -1 + height for coord in path]
        if distance <= 5:  # 50 meters (5 in 10x10 meter grid)
            plt.plot(x_coords, y_coords, color=tuple(col), alpha=0.6)
            cluster_start_points_k += 1
        else:
            plt.plot(x_coords, y_coords, 'black', alpha=0.01)

    cluster_start_points += cluster_start_points_k
    total_start_points += len(paths)
    print(f"Cluster {k}: Representative Point: {representative_point}, Streams: {cluster_start_points_k * grid_size}")

# Calculate the area of the land
land_area = np.sum(image_array != 0) * 100  # Each point represents 10x10 meters (100 square meters)
coverage_percentage = (cluster_start_points / total_start_points) * 100
print(f"Land area: {land_area} square meters")
print(f"Coverage percentage of start points: {coverage_percentage:.2f}%")

plt.title("Paths with K-Means Clustering, Alpha Shapes, and Representative End Points")
plt.legend()
plt.show()
