from rasterio.crs import CRS
import os
import rasterio
from pyproj import CRS, Transformer

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)
    image = dataset.read()

    # Define the desired dimensions
    target_width, target_height = 5952, 3040

    # Create a new directory to save the modified images
    output_dir = 'C:/Users/LENOVO/Desktop/thesis/new_vector/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")  # Debugging statement

    # Read the subset of the image (first 5952x3040 pixels from the top-left corner)
    subset = image[:, :target_height, :target_width]  # Corrected dimensions

    # Save the subset with the same filename in the output directory
    output_file = os.path.join(output_dir, os.path.basename(tif_file))
    print(f"Output File: {output_file}")  # Debugging statement
    with rasterio.open(
        output_file,
        'w',
        driver='PNG',
        width=subset.shape[2],
        height=subset.shape[1],
        count=subset.shape[0],  # Set count to the number of bands in the subset
        dtype=subset.dtype
    ) as dst:
        dst.write(subset)

if __name__ == '__main__':
    piece_info = []  # Initialize piece_info list
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/vector/'
    for file in os.listdir(path):
        if file.endswith('.png'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        print(f"Processing file: {tif_file}")  # Debugging statement
        raster_extraction(index, tif_file)

    print('Finished successfully')
