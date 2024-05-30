import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # This avoids issues with large images
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

def load_image_as_array(file_path):
    with Image.open(file_path) as img:
        img = img.convert('L')  # Convert to grayscale
        img_array = np.array(img)
        img_array[img_array < 0] = 0  # Set negative values to zero
    return img_array

def find_steepest_descent(image_array, start_pos, max_radius=25):
    x, y = start_pos
    path = [(x, y)]
    while True:
        current_intensity = image_array[x, y]
        if current_intensity == 0:
            break

        min_intensity = current_intensity
        move = (0, 0)
        moved = False

        for radius in range(1, max_radius + 1):
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
                break
        if not moved:
            break
    return path

def process_paths(args):
    image_array, positions, max_radius = args
    return [find_steepest_descent(image_array, pos, max_radius) for pos in positions]

def cluster_and_plot(image_array, grid_size, max_radius=25):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='jet')

    grid_x = np.linspace(0, image_array.shape[0] - 1, grid_size)
    grid_y = np.linspace(0, image_array.shape[1] - 1, grid_size)
    start_positions = [(int(x), int(y)) for x in grid_x for y in grid_y]

    chunks = [start_positions[i:i+10] for i in range(0, len(start_positions), 10)]  # Chunking positions

    with ProcessPoolExecutor() as executor:
        args = [(image_array, chunk, max_radius) for chunk in chunks]
        results = executor.map(process_paths, args)

    paths = [path for result in results for path in result if len(path) > 1]
    endpoints = [path[-1] for path in paths]
    start_positions = [path[0] for path in paths]

    if endpoints:
        endpoints = np.array(endpoints)
        clusters = cluster_endpoints(endpoints)
        plot_clusters(image_array, clusters, paths, start_positions, ax)

    ax.axis('off')
    plt.show()

# Load image and execute
image_array = load_image_as_array('FO_DSM_2017_FOTM_10M.tif')
cluster_and_plot(image_array, 20)