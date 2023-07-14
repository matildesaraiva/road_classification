# Some imports
import os
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import Affine

def raster_extraction(index,jp2_file):
    print(f'{index} - {jp2_file}')
    dataset = rasterio.open(jp2_file)

    # Convert the CRS
    # standard WGS84 coordinates
    new_crs = CRS.from_epsg(4326)
    # Figuring out the top left coordinates in WGS84 system
    topleft_coo = transform(dataset.crs, new_crs,
                        xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
    # Figuring out the bottom right coordinates in WGS84 system
    bottomright_coo = transform(dataset.crs, new_crs,
                        xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])
    # Read the image data
    image = dataset.read()
    # Get the dimensions of the image
    height, width = image.shape[1:]
    # Get the geospatial information
    transform_dataset = dataset.transform
    crs = dataset.crs
    # Create a list to store the coordinates and saved file paths
    piece_info = []
    #pieces = []
    # Split the image into 3x3 grid
    for i in range(30):
        for j in range(30):
            # Calculate the window bounds for each piece
            start_h = i * height // 30
            end_h = (i + 1) * height // 30
            start_w = j * width // 30
            end_w = (j + 1) * width // 30
            # Read the subset of the image using window
            window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
            subset = dataset.read(window=window)
            # Create a new TIFF file for each piece
            piece_path = f"C:/data/raster_data/raster_{index}_{i}_{j}.tif"
            with rasterio.open(
                    piece_path,
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
        piece_info.append((piece_coords, piece_path))
        # Display the pieces
        #for piece in pieces:
            #rp.show(piece.squeeze())
        # Print the piece coordinates and file paths
        for info in piece_info:
            print(f"Coordinates: {info[0]}")
            print(f"File path: {info[1]}")

if __name__ == '__main__':
    raster_files = []
    path = 'C:/data/original_raster'
    for file in os.listdir(path):
        if file.endswith('.jp2'):
            jp2_file = os.path.join(path, file)
            raster_files.append(jp2_file)
    for index, jp2_file in enumerate(raster_files):
        raster_extraction(index, jp2_file)
    print('Finished successfully')