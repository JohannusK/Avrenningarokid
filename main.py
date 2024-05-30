import numpy as np
import datashader as ds
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt


def load_image_as_array(file_path):
    with Image.open(file_path) as img:
        # Convert image to grayscale for intensity processing
        #img = img.convert('L')
        #img_array = np.array(img)
        img_array = np.array(img)[2682:3062, 4350:4776]
        #img_array = np.array(img)[5364:6124, 8700:9552]
        img_array[img_array < 0] = 0
    return img_array


def find_steepest_descent(image_array, start_pos, max_radius=25):
    x, y = start_pos
    path = [(x, y)]

    while True:
        current_intensity = image_array[x, y]
        if current_intensity == 0:
            break

        min_intensity = current_intensity
        move = (0, 0)  # (dx, dy)
        moved = False

        # Check in increasing radius shells until a move is made or max radius is reached
        for radius in range(5, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < image_array.shape[0] and 0 <= ny < image_array.shape[1]:
                            neighbor_intensity = image_array[nx, ny]
                            if neighbor_intensity < min_intensity:
                                min_intensity = neighbor_intensity
                                move = (dx, dy)
            if move != (0, 0):
                x += move[0]
                y += move[1]
                path.append((x, y))
                moved = True
                break  # Break the radius loop if a move is made

        if not moved:
            break  # Stop if no move was made within the maximum radius

    return path


def descent_wrapper(args):
    return find_steepest_descent(*args)


def cluster_endpoints(endpoints, threshold=20, min_cluster_size=10):
    # Calculate pairwise distances between endpoints
    distances = cdist(endpoints, endpoints)
    # Find clusters where endpoints are within the threshold distance
    clusters = []
    visited = set()

    for i in range(len(endpoints)):
        if i not in visited:
            # Find all points within the threshold distance of point i
            cluster = np.where(distances[i] <= threshold)[0]
            visited.update(cluster)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
    return clusters


def plot_clusters(image_array, clusters, paths, start_positions, ax):
    # Sort clusters by size (descending order) and get the top 10
    sorted_clusters = sorted(clusters, key=len, reverse=True)[:50]

    # Define a color map with 10 unique colors
    colors = plt.cm.get_cmap('tab20', len(sorted_clusters))

    for i, cluster in enumerate(sorted_clusters):
        for idx in cluster:
            x_coords, y_coords = zip(*paths[idx])
            ax.plot(y_coords, x_coords, color=colors(i), marker='.', markersize=0.5, linestyle='-', alpha=0.7)
            # Mark the start position with a red dot
            start_x, start_y = start_positions[idx]
            ax.plot(start_y, start_x, color=colors(i), markersize=3)


def plot_all_paths(image_array, grid_size):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='jet')

    # Generate the grid of start positions
    grid_x = np.linspace(0, image_array.shape[0] - 1, grid_size)
    grid_y = np.linspace(0, image_array.shape[1] - 1, grid_size)

    endpoints = []
    paths = []
    start_positions = []

    for x in grid_x:
        for y in grid_y:
            start_pos = (int(x), int(y))
            path = find_steepest_descent(image_array, start_pos, max_radius=40)
            if len(path) > 1:
                paths.append(path)
                endpoints.append(path[-1])  # Store the endpoint of each path
                start_positions.append((x, y))  # Store the start position

    # Cluster endpoints
    endpoints = np.array(endpoints)
    clusters = cluster_endpoints(endpoints)

    # Plot only the top 10 largest clustered paths
    plot_clusters(image_array, clusters, paths, start_positions, ax)

    ax.axis('off')
    plt.show()

image_array = load_image_as_array('FO_DSM_2017_FOTM_10M.tif')

# Plot all paths from a 100x100 grid
plot_all_paths(image_array, 100)