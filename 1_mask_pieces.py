# Description of the notebook:
# Vector's cut: Split each image into 32x32 pixel pieces

from rasterio.windows import Window
import os
import rasterio
import numpy as np

input_path = 'C:/Users/LENOVO/Desktop/thesis/vector/'
output_path_zero = 'C:/Users/LENOVO/Desktop/thesis/vector_pieces/no_road/'
output_path_nonzero = 'C:/Users/LENOVO/Desktop/thesis/vector_pieces/road/'

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        print(png_file)
        dataset = rasterio.open(png_file)
        # Read the image data
        image = dataset.read()
        # Get the dimensions of the image
        height, width = image.shape[1:]
        for i in range(96):
            for j in range(186):
                # Calculate the window bounds for each piece
                start_h = int(i * height / 96)
                end_h = int((i + 1) * height / 96)
                start_w = int(j * width / 186)
                end_w = int((j + 1) * width / 186)

                # Read the subset of the image using window
                window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
                subset = dataset.read(window=window)

                # Check if all values in the subset are equal to 0
                if np.all(subset == 0):
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_zero, f'{identifier}_{i}_{j}.tif')
                else:
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_nonzero, f'{identifier}_{i}_{j}.tif')

                with rasterio.open(output_path, 'w', driver='GTiff', width=subset.shape[2], height=subset.shape[1], count=1, dtype=subset.dtype) as dst:
                    dst.write(subset)