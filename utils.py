from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd
import datashader as ds
from datashader import transfer_functions as tf
from matplotlib import pyplot as plt

def plotFigure(img_pil, image_array):
    # Display the image and add contour lines
    plt.figure(figsize=(10, 6))
    plt.imshow(img_pil, aspect='equal')
    contour_levels = [0, 100, 150, 200, 250, 300]
    plt.contour(image_array, levels=contour_levels, colors='black', linewidths=1, origin='image')
    plt.gca().invert_yaxis()  # Adjust axis if necessary depending on data orientation
    plt.axis('off')  # Turn off the axis

def loadMap(file_path, bounds):

    # Load the image with transformed bounds
    image_array = load_transformed_image(file_path, bounds)
    height, width = image_array.shape
    # Prepare DataFrame for datashader
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    df = pd.DataFrame({
        'x': x_coords.flatten(),
        'y': y_coords.flatten(),
        'intensity': image_array.flatten()
    })

    # Set up datashader canvas and aggregate data
    cvs = ds.Canvas(plot_width=width, plot_height=height, x_range=(0, width), y_range=(0, height))
    agg = cvs.points(df, 'x', 'y', ds.mean('intensity'))

    # Generate image using 'terrain' colormap
    img = tf.shade(agg, cmap=plt.cm.terrain, how='linear')
    img_pil = img.to_pil()
    return image_array, img_pil
def transform_coords(lat, lon, target_crs='EPSG:5316'):
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return transformer.transform(lon, lat)


def load_transformed_image(file_path, bounds):
    left, bottom = transform_coords(bounds[1], bounds[0])
    right, top = transform_coords(bounds[3], bounds[2])
    transformed_bounds = (left, bottom, right, top)

    with rasterio.open(file_path) as src:
        window = from_bounds(left, bottom, right, top, src.transform)
        img_array = src.read(1, window=window)
        if src.nodata is not None:
            img_array = np.where(img_array == src.nodata, 0, img_array)
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
