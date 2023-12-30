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
output_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/no_border/no_road/'
output_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth//no_border/road/'
output_excess_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/no_border/no_road/'
output_excess_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/no_border/road/'

all_zero_pieces = []
road_dataset_pieces = []
road_excess_pieces = []

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        image = cv2.imread(png_file)

        height, width, _ = image.shape
        height_pieces = int(height // 32)
        width_pieces = int(width // 32)

        for i in range(height_pieces):
            for j in range(width_pieces):
                # Calculate the window bounds for each piece
                start_h = int(i * height // height_pieces)
                end_h = int((i + 1) * height // height_pieces)
                start_w = int(j * width // width_pieces)
                end_w = int((j + 1) * width // width_pieces)

                subset = image[start_h:end_h, start_w:end_w, :]

                if np.all(subset == 0):
                    identifier = os.path.basename(png_file).split(".png")[0]
                    piece_name = f"{identifier}_{i}_{j}.png"
                    all_zero_pieces.append((piece_name, subset))
                else:
                    non_zero = np.count_nonzero(subset)
                    total = subset.size
                    portion = non_zero / total

                    if portion >= 0.2:
                        identifier = os.path.basename(png_file).split(".png")[0]
                        piece_name = f"{identifier}_{i}_{j}.png"
                        road_dataset_pieces.append((piece_name, subset))
                    else:
                        identifier = os.path.basename(png_file).split(".png")[0]
                        piece_name = f"{identifier}_{i}_{j}.png"
                        road_excess_pieces.append((piece_name, subset))

# print do tamanho da lista de todos os No Road sem shuffle
len_all_zero_pieces = len(all_zero_pieces)
print(len_all_zero_pieces)
# print do tamanho da lista dos Road para o dataset
len_road_dataset_pieces = len(road_dataset_pieces)
print(len_road_dataset_pieces)
# print do tamanho da lista dos Road para o excess
len_road_excess_pieces = len(road_excess_pieces)
print(len_road_excess_pieces)
# Shuffle da lista de todos os No Road
random.shuffle(all_zero_pieces)
# Split all_zero_pieces into two lists
zero_dataset_pieces = [(piece_name, subset.copy()) for piece_name, subset in all_zero_pieces[:len(road_dataset_pieces)]]
zero_excess_pieces = [(piece_name, subset.copy()) for piece_name, subset in all_zero_pieces[len(road_dataset_pieces):]]
print(len(zero_dataset_pieces))
print(len(zero_excess_pieces))

# Save zero_dataset_pieces in dataset/groundtruth/no road
for piece_name, subset in zero_dataset_pieces:
    output_path = os.path.join(output_balanced_no_road, piece_name)
    cv2.imwrite(output_path, subset)

# Save road_dataset_pieces in dataset/groundtruth/road
for piece_name, subset in road_dataset_pieces:
    output_path = os.path.join(output_balanced_road, piece_name)
    cv2.imwrite(output_path, subset)

# Save zero_excess_pieces in excess/groundtruth/no road
#for piece_name, subset in zero_excess_pieces:
#    output_path = os.path.join(output_excess_no_road, piece_name)
#    cv2.imwrite(output_path, subset)

# Save road_excess_pieces in excess/groundtruth/road
#for piece_name, subset in road_excess_pieces:
#    output_path = os.path.join(output_excess_road, piece_name)
#    cv2.imwrite(output_path, subset)