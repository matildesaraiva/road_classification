# Some imports
import os
from rasterio.crs import CRS
from rasterio.warp import transform
import rasterio
from rasterio.features import geometry_mask
import osmnx as ox
import networkx as nx
from rasterio.transform import from_origin


# Import the raster data with rasterio
path = 'C:/data/raster_data/'
for file in os.listdir(path):
    if file.endswith('.tif'):
        tif_file = os.path.join(path, file)
        print(tif_file)
        raster = rasterio.open(tif_file)

        new_crs = CRS.from_epsg(4326)
        # Figuring out the top left coordinates in WGS84 system
        topleft_coo = transform(raster.crs, new_crs, xs=[raster.bounds[0]], ys=[raster.bounds[3]])
        bottomright_coo = transform(raster.crs, new_crs, xs=[raster.bounds[2]], ys=[raster.bounds[1]])

        north = topleft_coo[1][0]  # northern latitude
        south = bottomright_coo[1][0]  # southern latitude
        east = bottomright_coo[0][0]  # eastern longitude
        west = topleft_coo[0][0]  # western longitude

        try:
            # Download the OSM data and create a GeoDataFrame
            graph = ox.graph.graph_from_bbox(north, south, east, west, network_type='all', retain_all=True)
        except ValueError as e:
            if "Found no graph nodes within the requested polygon" in str(e):
                print("No graph nodes found within the requested polygon for:", tif_file)
            else:
                print("ValueError occurred for:", tif_file)
            continue

        components = nx.strongly_connected_components(graph)
        for i, component in enumerate(components):
            subgraph = graph.subgraph(component)
            try:
                geodf = ox.utils_graph.graph_to_gdfs(subgraph, nodes=True, edges=True, node_geometry=True,
                                                     fill_edge_geometry=True)
            except ValueError as e:
                if "graph contains no edges" in str(e):
                    print("Graph contains no edges for:", tif_file)
                else:
                    print("ValueError occurred for:", tif_file)
                break
            gdf = geodf[1]

            file_name = tif_file.split("raster")[-1]
            output_path = f"C:/data/vector_data/vector{file_name}"

            # Define the desired raster resolution and extent
            xmin, ymin, xmax, ymax = west, south, east, north
            width = int(raster.meta["width"])
            height = int(raster.meta["height"])
            x_pixel_size = ((xmax - xmin) / width)
            y_pixel_size = ((ymax - ymin) / height)
            affine_transform = from_origin(xmin, ymax, x_pixel_size, y_pixel_size)

            new_dataset = rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='uint8',
                crs=gdf.crs,
                transform=affine_transform
            )

            mask = geometry_mask(gdf.geometry, out_shape=(height, width), transform=affine_transform, invert=True)
            new_dataset.write(mask, 1)
            new_dataset.close()


