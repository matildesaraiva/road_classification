# Some imports
import os
from rasterio.crs import CRS
from rasterio.warp import transform
import rasterio
from rasterio.features import geometry_mask
import osmnx as ox
import networkx as nx
from rasterio.transform import from_origin

raster = rasterio.open('C:/Users/LENOVO/Desktop/thesis/pieces/9_1_1.tif')
new_crs = CRS.from_epsg(4326)
# Figuring out the top left coordinates in WGS84 system
topleft_coo = transform(raster.crs, new_crs, xs=[raster.bounds[0]], ys=[raster.bounds[3]])
bottomright_coo = transform(raster.crs, new_crs, xs=[raster.bounds[2]], ys=[raster.bounds[1]])
north = topleft_coo[1][0]  # northern latitude
south = bottomright_coo[1][0]  # southern latitude
east = bottomright_coo[0][0]  # eastern longitude
west = topleft_coo[0][0]  # western longitude
print(north)
print(south)
print(east)
print(west)