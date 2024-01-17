# Description of the notebook: Vector's cut: Split each image into 32x32 pixel pieces and save them in 4 different folders:
##### Balanced dataset:
##'03_balanced_groundtruth/no_road/': contains files that do not contain roads in the same number of files as the other class;
##'03_balanced_groundtruth/road/': contains files that contain roads in more than 20% of the total number of pixels, in the same quantity as the other class;
#### Excess data:
##'05_excess/groundtruth/no_road/': contains the remaining files that do not contain roads;
##05_excess/groundtruth/road/': contains the files that contain roads in less than 20% of the total number of pixels.

import os
import cv2
import numpy as np
import random

input_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/groundtruth/binary/border'
output_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium/groundtruth/border/no_road/'
output_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium/groundtruth/border/road/'
excess = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium/groundtruth/border_excess/'

excess_pieces = []
road_pieces = []
no_road_pieces = []

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        image = cv2.imread(png_file)

        height, width, _ = image.shape
        piece_size = 64

        # Calculate the number of pieces in each dimension
        height_pieces = height // piece_size
        width_pieces = width // piece_size

        for i in range(height_pieces):
            for j in range(width_pieces):
                # Calculate the window bounds for each piece
                start_h = i * piece_size
                end_h = (i + 1) * piece_size
                start_w = j * piece_size
                end_w = (j + 1) * piece_size

                # Ensure the subset has 96x96 pixels in height and width
                subset = image[start_h:end_h, start_w:end_w, :]

                if subset.shape[0] == piece_size and subset.shape[1] == piece_size:
                    if np.all(subset == 0):
                        identifier = os.path.basename(png_file).split(".png")[0]
                        piece_name = f"{identifier}_{i}_{j}.png"
                        no_road_pieces.append((piece_name, subset))
                    else:
                        non_zero = np.count_nonzero(subset)
                        total = subset.size
                        portion = non_zero / total
                        if portion >= 0.2:
                            identifier = os.path.basename(png_file).split(".png")[0]
                            piece_name = f"{identifier}_{i}_{j}.png"
                            road_pieces.append((piece_name, subset))
                        else:
                            identifier = os.path.basename(png_file).split(".png")[0]
                            piece_name = f"{identifier}_{i}_{j}.png"
                            excess_pieces.append((piece_name, subset))

len_no_road = len(no_road_pieces)
print(len_no_road)

len_road = len(road_pieces)
print(len_road)

len_excess = len(excess_pieces)
print(len_excess)

random.shuffle(no_road_pieces)

for piece_name, subset in no_road_pieces[:len(road_pieces)]:
    output_path = os.path.join(output_no_road, piece_name)
    cv2.imwrite(output_path, subset)

for piece_name, subset in no_road_pieces[len(road_pieces):]:
    output_path = os.path.join(excess, piece_name)
    cv2.imwrite(output_path, subset)

for piece_name, subset in road_pieces:
    output_path = os.path.join(output_road, piece_name)
    cv2.imwrite(output_path, subset)

for piece_name, subset in excess_pieces:
    output_path = os.path.join(excess, piece_name)
    cv2.imwrite(output_path, subset)


