# Description of the notebook:
# Raster's second cut: Split each image into 32x32 pixel pieces

from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
import os
import rasterio
from rasterio.transform import Affine
from pyproj import CRS

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)
    # Convert the CRS to standard WGS84 coordinates
    new_crs = CRS.from_epsg(4326)
    topleft_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
    bottomright_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])
    # Read the image data
    image = dataset.read()
    # Get the dimensions of the image
    height, width = image.shape[1:]
    # Get the geospatial information
    transform_dataset = dataset.transform
    crs = dataset.crs
    # Create a list to store the coordinates and saved file paths
    piece_info = []

    # Split the image into 95x186 grid
    for i in range(95):
        for j in range(186):
            # Calculate the window bounds for each piece
            start_h = int(i * height / 95)
            end_h = int((i + 1) * height / 95)
            start_w = int(j * width / 186)
            end_w = int((j + 1) * width / 186)
            # Read the subset of the image using window
            window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
            subset = dataset.read(window=window)
            # Create a new TIFF file for each piece
            identifier = os.path.basename(tif_file).split(".tif")[0]
            piece_name = f"{identifier}_{i}_{j}.tif"
            # Vector folders to separate the pieces into categories (different directories)
            no_road_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/no_road/'
            road_center_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_center/'
            road_other_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_other/'
            # Condition for the attribution of file to each folder
            if os.path.exists(os.path.join(no_road_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/no_road/{piece_name}'
            elif os.path.exists(os.path.join(road_center_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_center/{piece_name}'
            elif os.path.exists(os.path.join(road_other_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_other/{piece_name}'
            else:
                output_path = None
            if output_path:
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=subset.shape[1],
                    width=subset.shape[2],
                    count=subset.shape[0],
                    dtype=subset.dtype,
                    crs=crs,
                    transform=transform_dataset * Affine.translation(start_w, start_h)
                ) as dst:
                    # Write the subset to the TIFF file
                    dst.write(subset)
            # Calculate the geographical coordinates of the piece
            piece_coords = transform_dataset * (start_w, start_h)
            piece_info.append((piece_coords, output_path))
    print(output_path)

if __name__ == '__main__':
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/raster/'
    for file in os.listdir(path):
        if file.endswith('.tif'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        raster_extraction(index, tif_file)

    print('Finished successfully')
