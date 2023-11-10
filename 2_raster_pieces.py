# Description of the notebook:
# Raster's second cut: Split each image into 32x32 pixel pieces

from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
import os
import rasterio
from rasterio.transform import Affine
from pyproj import CRS

import cv2
import os

def split_and_save_image(input_path):
    for file in os.listdir(input_path):
        if file.endswith('.tif'):
            tif_file = os.path.join(input_path, file)
            print(tif_file)
            # Read the image using OpenCV
            image = cv2.imread(tif_file)
            # Get the dimensions of the image
            height, width, _ = image.shape
            for i in range(95):
                for j in range(186):
                    # Calculate the window bounds for each piece #####ADICIONAR EM VARIÁVEIS E NÃO SÓ NUMERO#########
                    start_h = int(i * height / 95)
                    end_h = int((i + 1) * height / 95)
                    start_w = int(j * width / 186)
                    end_w = int((j + 1) * width / 186)
                    # Extract the subset of the image
                    subset = image[start_h:end_h, start_w:end_w, :]
                    # Create a new PNG file for each piece
                    identifier = os.path.basename(tif_file).split(".tif")[0]
                    piece_name = f"{identifier}_{i}_{j}.png"
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
                        # Save the subset as a new image
                        cv2.imwrite(output_path, subset)

    print('Finished successfully')

if __name__ == '__main__':
    input_path = 'C:/Users/LENOVO/Desktop/thesis/raster/'
    split_and_save_image(input_path)
