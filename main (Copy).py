import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt


def load_image_as_array(file_path):
    with Image.open(file_path) as img:
        # Convert image to grayscale for intensity processing
        #img = img.convert('L')
        #img_array = np.array(img)[2682:3062, 4350:4776]
        img_array = np.array(img)[5364:6124, 8700:9552]
        img_array[img_array < 0] = 0
    return img_array


def find_steepest_descent(image_array, start_pos, max_radius=20):
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


def plot_all_paths(image_array, grid_size):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_array, cmap='jet')

    # Generate the grid of start positions
    grid_x = np.linspace(0, image_array.shape[0] - 1, grid_size)
    grid_y = np.linspace(0, image_array.shape[1] - 1, grid_size)
    start_positions = [(image_array, (int(x), int(y)), 25) for x in grid_x for y in grid_y]

    # Create a process pool executor to use all available cores
    with ProcessPoolExecutor() as executor:
        results = executor.map(descent_wrapper, start_positions)

    # Process results and plot
    for path in results:
        if len(path) > 1:
            x_coords, y_coords = zip(*path)
            ax.plot(y_coords, x_coords, color='black', marker='.', markersize=0.5, linestyle='-', alpha=0.005)

    ax.axis('off')
    plt.show()


image_array = load_image_as_array('FO_DSM_2017_FOTM_5M.tif')

# Plot all paths from a 100x100 grid
plot_all_paths(image_array, 200)