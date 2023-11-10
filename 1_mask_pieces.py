# Description of the notebook: Vector's cut: Split each image into 32x32 pixel pieces

import os
import cv2
import numpy as np

input_path = 'C:/Users/LENOVO/Desktop/thesis/groundtruth/'
output_path_no_road = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/no_road/'
output_path_road_center = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_center/'
output_path_road_other = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_other/'

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        print(png_file)
        # Read the image using OpenCV
        image = cv2.imread(png_file)
        # Get the dimensions of the image
        height, width, _ = image.shape
        for i in range(95):
            for j in range(186):
                # Calculate the window bounds for each piece
                start_h = int(i * height / 95)
                end_h = int((i + 1) * height / 95)
                start_w = int(j * width / 186)
                end_w = int((j + 1) * width / 186)
                # Extract the subset of the image
                subset = image[start_h:end_h, start_w:end_w, :]
                # Check if all values in the subset are equal to 0
                if np.all(subset == 0):
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_no_road, f'{identifier}_{i}_{j}.png')
                else:
                    # Check if values are different from 0 in the center of the piece
                    center_h = int(subset.shape[0] / 2)
                    center_w = int(subset.shape[1] / 2)
                    if subset[center_h, center_w, 0] != 0:
                        identifier = os.path.basename(file).split(".png")[0]
                        output_path = os.path.join(output_path_road_center, f'{identifier}_{i}_{j}.png')
                    else:
                        identifier = os.path.basename(file).split(".png")[0]
                        output_path = os.path.join(output_path_road_other, f'{identifier}_{i}_{j}.png')
                # Save the subset as a new image
                cv2.imwrite(output_path, subset)