# Some imports
import os
import rasterio
from rasterio.crs import CRS
import rasterio.plot as rp
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import Affine
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.plot import show
import cv2
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import osmnx as ox
import networkx as nx

path = 'C:/data/original'
for file in os.listdir(path):
    if file.endswith('.jp2'):
        jp2_file = os.path.join(path, file)
        print(jp2_file)
        dataset = rasterio.open(jp2_file)

#rp.show(dataset)
#print(dataset.meta)
#print(dataset.bounds)

# Convert the CRS
# standard WGS84 coordinates
new_crs = CRS.from_epsg(4326)
# Figuring out the top left coordinates in WGS84 system
topleft_coo = transform(dataset.crs, new_crs,
                    xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
# Figuring out the bottom right coordinates in WGS84 system
bottomright_coo = transform(dataset.crs, new_crs,
                    xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])
print("Top-left coordinates (long, lat):", topleft_coo)
print("Bottom-right coordinates (long, lat):", bottomright_coo)

# Read the image data
image = dataset.read()

# Get the dimensions of the image
height, width = image.shape[1:]

# Get the geospatial information
transform_dataset = dataset.transform
crs = dataset.crs

# Create a list to store the coordinates and saved file paths
piece_info = []
pieces = []

# Split the image into 30x30 grid
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
        piece_path = f"C:/data/pieces/raster_{i}_{j}.tif"
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
#for info in piece_info:
    #print(f"Coordinates: {info[0]}")
    #print(f"File path: {info[1]}")

# Import the raster data with rasterio
path = 'C:/data/pieces'
for file in os.listdir(path):
    if file.endswith('.tif'):
        tif_file = os.path.join(path, file)
        print(tif_file)
        dataset = rasterio.open(tif_file)

#show(raster)
#print(raster.meta)
#print(raster.bounds)

# standard WGS84 coordinates
new_crs = CRS.from_epsg(4326)
# Figuring out the top left coordinates in WGS84 system
topleft_coo = transform(raster.crs, new_crs,
                    xs=[raster.bounds[0]], ys=[raster.bounds[3]])
# Figuring out the bottom right coordinates in WGS84 system
bottomright_coo = transform(raster.crs, new_crs,
                    xs=[raster.bounds[2]], ys=[raster.bounds[1]])

# Define the bounding box coordinates
north = topleft_coo[1][0]  # northern latitude
south = bottomright_coo[1][0]  # southern latitude
east = bottomright_coo[0][0]  # eastern longitude
west = topleft_coo[0][0]  # western longitude

#print(north)
#print(south)
#print(east)
#print(west)

# Download the OSM data and create a graph
graph = ox.graph.graph_from_bbox(north, south, east, west, network_type='all')

# Get the connected components
components = nx.strongly_connected_components(graph)

for i, component in enumerate(components):
    subgraph = graph.subgraph(component)
    geodf = ox.utils_graph.graph_to_gdfs(subgraph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

gdf = geodf[1]
#print(gdf)

# Define the desired raster resolution and extent
xmin, ymin, xmax, ymax = west, south, east, north
width = int(raster.meta["width"])
height = int(raster.meta["height"])
x_pixel_size = ((xmax - xmin) / width)
y_pixel_size = ((ymax - ymin) / height)
transform = from_origin(xmin, ymax, x_pixel_size, y_pixel_size)

# Create a new raster dataset
vector_raster_path = 'C:/data/vector_raster.tif'
new_dataset = rasterio.open(
    vector_raster_path,
    'r+',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype='uint8',
    crs=gdf.crs,
    transform=transform)

#mask = geometry_mask(gdf.geometry, out_shape=(height, width), transform=transform, invert=True)
# Write the mask to the raster dataset
#new_dataset.write(mask, 1)

# Open vec as a numpy array
image = new_dataset.read()
new_dataset.close()
# Create a copy of the image
modified_image = np.copy(image)

for row in range(image.shape[1]):
    for col in range(1, image.shape[2] - 1):
        pixel_value = image[0, row, col]
        if pixel_value == 1:
            modified_image[0, row, col - 1] = 1
            #For now, we only add 1 to the pixel before, so that the line isn't too large (test)
            #modified_image[0, row, col + 1] = 1

output_path = 'C:/data/vector.tif'
with rasterio.open(
    output_path,
    'r+',
    driver='GTiff',
    height=modified_image.shape[1],
    width=modified_image.shape[2],
    count=1,
    dtype=modified_image.dtype,
    crs=rasterio.crs.CRS.from_epsg(4326),
    transform=raster.meta['transform']) as dst:
    # Write the modified_image array to the raster dataset
    dst.write(modified_image)

# This part is going to be deleted, it only has the purpose of making it easier to access if both images match
background = cv2.imread('piece_0_6.tif')
overlay = cv2.imread('C:/data/vector.tif')
# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)
# Save the combined image
cv2.imwrite('C:/data/combined.tif', combined_image)