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

input_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/groundtruth/binary/no_border'
output_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium_10/groundtruth/no_road/'
output_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium_10/groundtruth/road/'

all_zero_pieces = []
road_dataset_pieces = []
road_excess_pieces = []

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
                        all_zero_pieces.append((piece_name, subset))
                    else:
                        non_zero = np.count_nonzero(subset)
                        total = subset.size
                        portion = non_zero / total

                        if portion >= 0.1:
                            identifier = os.path.basename(png_file).split(".png")[0]
                            piece_name = f"{identifier}_{i}_{j}.png"
                            road_dataset_pieces.append((piece_name, subset))
                        else:
                            continue

# print do tamanho da lista de todos os No Road sem shuffle
len_all_zero_pieces = len(all_zero_pieces)
print(len_all_zero_pieces)
# print do tamanho da lista dos Road para o dataset
len_road_dataset_pieces = len(road_dataset_pieces)
print(len_road_dataset_pieces)
# Shuffle da lista de todos os No Road
random.shuffle(all_zero_pieces)

# Save zero_dataset_pieces in dataset/groundtruth/no road
for piece_name, subset in all_zero_pieces[:len(road_dataset_pieces)]:
    output_path = os.path.join(output_balanced_no_road, piece_name)
    cv2.imwrite(output_path, subset)

# Save road_dataset_pieces in dataset/groundtruth/road
for piece_name, subset in road_dataset_pieces:
    output_path = os.path.join(output_balanced_road, piece_name)
    cv2.imwrite(output_path, subset)