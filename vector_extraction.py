# some imports
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.plot import show
import osmnx
import matplotlib.pyplot as plt
import cv2
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import networkx as nx

raster = rasterio.open('piece_0_5.tif')
#show(raster)
print(raster.meta)
print(raster.bounds)

# Convert the CRS
# standard WGS84 coordinates
new_crs = CRS.from_epsg(4326)
# Figuring out the top left coordinates in WGS84 system
topleft_coo = transform(raster.crs, new_crs,
                    xs=[raster.bounds[0]], ys=[raster.bounds[3]])
# Figuring out the bottom right coordinates in WGS84 system
bottomright_coo = transform(raster.crs, new_crs,
                    xs=[raster.bounds[2]], ys=[raster.bounds[1]])

#Define the bounding box coordinates
north = topleft_coo[1][0]  # northern latitude
south = bottomright_coo[1][0]  # southern latitude
east = bottomright_coo[0][0]  # eastern longitude
west = topleft_coo[0][0]  # western longitude
#Download the OSM data and create a GeoDataFrame
graph = osmnx.graph.graph_from_bbox(north,
                                south,
                                east,
                                west,
                                network_type='all',
                                retain_all=True
                               )
# Get the connected components
components = nx.strongly_connected_components(graph)
subgraphs = [graph.subgraph(component) for component in components]
# Specify the output geopackage file path
output_path = "C:/Users/LENOVO/Desktop/thesis/analysis/notebooks/data/geodf.gpkg"
osmnx.save_graph_geopackage(graph, filepath=output_path, encoding='utf-8', directed=False)

geodf = osmnx.utils_graph.graph_to_gdfs(graph
                                        nodes=True,
                                        edges=True,
                                        node_geometry=True,
                                        fill_edge_geometry=True)
# Specify the desired figure size in inches
fig_width = int(raster.meta["width"])
fig_height = int(raster.meta["height"])
# Create the figure with the specified size
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

# Plot the GeoDataFrame with red color
geodf[1].plot(ax=ax, color='red')  # Set the color to red

# Specify the range of the x and y axes
ax.set_xlim([west, east])  # Replace 'xmin' and 'xmax' with the desired x-axis range
ax.set_ylim([south, north])  # Replace 'ymin' and 'ymax' with the desired y-axis range

# Save the plot as a tif file with desired dimensions
output_file = 'plot.tif'  # Replace with your desired file name
plt.savefig(output_file, dpi=150)
# Display the plot
plt.show()

# Specify the output raster file path
output_raster_path = 'C:/data/output_raster.tif'
# Define the desired raster resolution and extent
xmin, ymin, xmax, ymax = west, south, east, north
width = int(raster.meta["width"])
height = int(raster.meta["height"])
x_pixel_size = ((xmax - xmin) / width)
y_pixel_size = ((ymax - ymin) / height)

transform = from_origin(xmin, ymax, x_pixel_size, y_pixel_size)

# Create a new raster dataset
new_dataset = rasterio.open(
    output_raster_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype='uint8',
    crs=gdf.crs,
    transform=transform
)

# Convert the GeoDataFrame to a mask
mask = geometry_mask(gdf.geometry, out_shape=(height, width), transform=transform, invert=True)

# Write the mask to the raster dataset
new_dataset.write(mask, 1)
new_dataset.close()

background = cv2.imread('piece_0_5.tif')
overlay = cv2.imread('C:/data/output_raster.tif')

# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]

# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)

# Save the combined image
cv2.imwrite('combined.png', combined_image)

background.shape
overlay.shape