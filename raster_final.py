import os
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import Affine
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon

path = '/content/drive/MyDrive/tese/Colab Notebooks/data/og_raster'
for file in os.listdir(path):
    if file.endswith('.jp2'):
        jp2_file = os.path.join(path, file)
        print(jp2_file)
        dataset = rasterio.open(jp2_file)
        # Convert the CRS
        # standard WGS84 coordinates
        new_crs = CRS.from_epsg(4326)
        # Figuring out the top left coordinates in WGS84 system
        topleft_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
        # Figuring out the bottom right coordinates in WGS84 system
        bottomright_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])
        north = topleft_coo[1][0]  # northern latitude
        south = bottomright_coo[1][0]  # southern latitude
        east = bottomright_coo[0][0]  # eastern longitude
        west = topleft_coo[0][0]  # western longitude
        # Read the image data
        image = dataset.read()
        # Get the dimensions of the image
        height, width = image.shape[1:]
        # Get the geospatial information
        transform_dataset = dataset.transform
        crs = dataset.crs
        # Create a list to store the coordinates and saved file paths
        piece_info = []
        # Split the image into 3x3 grid
        for i in range(43):
            for j in range(43):
                # Calculate the window bounds for each piece
                start_h = int(i * height / 42.890625)
                end_h = int((i + 1) * height / 42.890625)
                start_w = int(j * width / 42.890625)
                end_w = int((j + 1) * width / 42.890625)
                # Read the subset of the image using window
                window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
                subset = dataset.read(window=window)
                # count black pixels
                black_pixels = subset.size - np.count_nonzero(subset)
                if black_pixels > 10:
                    # Skip saving the image if it has more than 10 completely black pixels
                    continue
                try:
                    bbox = Polygon([(west, south), (east, south), (east, north), (west, north)])
                    place = "Angola"
                    gdf = ox.geocode_to_gdf(place)
                    angola_boundary = gdf.geometry.iloc[0]
                    angola_border = angola_boundary.intersection(bbox)
                    if angola_border.intersects(bbox):
                        identifier = os.path.basename(jp2_file).split(".jp2")[0]
                        piece_path = f"/content/drive/MyDrive/tese/Colab Notebooks/data/target_data/{identifier}_{i}_{j}.tif"
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
                    else:
                        # Create a new TIFF file for each piece
                        identifier = os.path.basename(jp2_file).split(".jp2")[0]
                        piece_path = f"/content/drive/MyDrive/tese/Colab Notebooks/data/raster_data/{identifier}_{i}_{j}.tif"
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
                except Exception as e:
                    print(f"Error processing piece ({i}, {j}): {str(e)}")