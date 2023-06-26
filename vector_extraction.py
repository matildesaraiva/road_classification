# some imports
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

# Import the raster data with rasterio
raster = rasterio.open('piece_0_6.tif')
print(raster)
#show(raster)
print(raster.meta)
print(raster.bounds)

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

# Download the OSM data and create a graph
graph = ox.graph.graph_from_bbox(north, south, east, west, network_type='all')

# Get the connected components
components = nx.strongly_connected_components(graph)

for i, component in enumerate(components):
    subgraph = graph.subgraph(component)
    geodf = ox.utils_graph.graph_to_gdfs(subgraph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)

gdf = geodf[1]
print(gdf)

# Define the desired raster resolution and extent
xmin, ymin, xmax, ymax = west, south, east, north
width = int(raster.meta["width"])
height = int(raster.meta["height"])
x_pixel_size = ((xmax - xmin) / width)
y_pixel_size = ((ymax - ymin) / height)
transform = from_origin(xmin, ymax, x_pixel_size, y_pixel_size)

# Create a new raster dataset
vector_raster_path = 'C:/data/vector_raster.tif' # vector_raster file path
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

mask = geometry_mask(gdf.geometry, out_shape=(height, width), transform=transform, invert=True)
# Write the mask to the raster dataset
new_dataset.write(mask, 1)

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

background = cv2.imread('piece_0_6.tif')
overlay = cv2.imread('C:/data/vector.tif')
# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)
# Save the combined image
cv2.imwrite('C:/data/combined.tif', combined_image)