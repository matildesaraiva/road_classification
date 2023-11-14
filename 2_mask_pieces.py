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

import os
import cv2
import numpy as np
import random

input_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/mask_groundtruth/'
output_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/no_road/'
output_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/road/'
output_excess_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/no_road/'
output_excess_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/road/'

all_zero_pieces = []

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        print(png_file)
        # Read the image using OpenCV
        image = cv2.imread(png_file)
        # Get the dimensions of the image
        height, width, _ = image.shape
        height_pieces = int(height // 32)
        width_pieces = int(width // 32)
        for i in range(height_pieces):
            for j in range(width_pieces):
                # Calculate the window bounds for each piece
                start_h = int(i * height / height_pieces)
                end_h = int((i + 1) * height / height_pieces)
                start_w = int(j * width / width_pieces)
                end_w = int((j + 1) * width / width_pieces)
                # Extract the subset of the image
                subset = image[start_h:end_h, start_w:end_w, :]

                # Check conditions for different subsets
                if np.all(subset == 0):
                    identifier = os.path.basename(png_file).split(".png")[0]
                    piece_name = f"{identifier}_{i}_{j}.png"
                    all_zero_pieces.append((piece_name, subset))
                else:
                    non_zero = np.count_nonzero(subset)
                    total = subset.size
                    portion = non_zero / total
                    # Check conditions for different subsets
                    if portion > 0.2:
                        identifier = os.path.basename(png_file).split(".png")[0]
                        piece_name = f"{identifier}_{i}_{j}.png"
                        output_path = os.path.join(output_balanced_road, piece_name)
                        cv2.imwrite(output_path, subset)
                    else:
                        identifier = os.path.basename(png_file).split(".png")[0]
                        piece_name = f"{identifier}_{i}_{j}.png"
                        output_path = os.path.join(output_excess_road, piece_name)
                        cv2.imwrite(output_path, subset)

# Shuffle the list of pieces containing only 0 valued pixels
random.shuffle(all_zero_pieces)

len_list = len(all_zero_pieces)
balancedr = len(os.listdir(output_balanced_road))

# Save pieces to output_balanced_no_road and output_excess_no_road
for item in range(len_list):
    piece_name, subset = all_zero_pieces[item]
    if item < balancedr:
        output_path = os.path.join(output_balanced_no_road, piece_name)
    else:
        output_path = os.path.join(output_excess_no_road, piece_name)
    cv2.imwrite(output_path, subset)
