from rasterio.crs import CRS
import os
import rasterio
from rasterio.transform import Affine
from pyproj import CRS, Transformer

# First cut: keep only the first 5952x3072 pixels starting in (0,0)

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)

    # Define the coordinate reference systems
    old_crs = dataset.crs
    new_crs = CRS.from_epsg(4326)

    # Create a transformer to perform the coordinate transformation
    transformer = Transformer.from_crs(old_crs, new_crs, always_xy=True)

    # Transform the coordinates
    topleft_coo = transformer.transform(dataset.bounds.left, dataset.bounds.top)
    bottomright_coo = transformer.transform(dataset.bounds.right, dataset.bounds.bottom)

    image = dataset.read()

    # Define the desired dimensions
    target_width, target_height = 5952, 3072

    # Define the transform for the new dataset
    transform_dataset = Affine.translation(topleft_coo[0], topleft_coo[1]) * Affine.scale(
        (bottomright_coo[0] - topleft_coo[0]) / target_width,
        (bottomright_coo[1] - topleft_coo[1]) / target_height
    )

    # Create a new directory to save the modified images
    output_dir = 'C:/Users/LENOVO/Desktop/thesis/original_raster/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")  # Debugging statement

    # Read the subset of the image (first 3072x5952 pixels)
    subset = image[:, :target_height, :target_width]  # Define 'subset' here before using it

    # Save the subset with the same filename in the output directory
    output_file = os.path.join(output_dir, os.path.basename(tif_file))
    print(f"Output File: {output_file}")  # Debugging statement
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=subset.shape[1],
        width=subset.shape[2],
        count=subset.shape[0],
        dtype=subset.dtype,
        crs=new_crs,  # Use the new CRS here
        transform=transform_dataset
    ) as dst:
        dst.write(subset)

if __name__ == '__main__':
    piece_info = []  # Initialize piece_info list
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/data/'
    for file in os.listdir(path):
        if file.endswith('.tif'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        print(f"Processing file: {tif_file}")  # Debugging statement
        raster_extraction(index, tif_file)

    print('Finished successfully')