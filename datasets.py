import os
import cv2
import numpy as np

input_path = 'C:/Users/LENOVO/Desktop/thesis/groundtruth/'
output_path_no_road = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/no_road/'
output_path_road = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road/'
output_path_do_not_use = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/do_not_use/'

# Define the threshold for considering an image as road
road_threshold = 0.2  # 20%

for file in os.listdir(input_path):
    if file.endswith('.png'):
        png_file = os.path.join(input_path, file)
        print(png_file)
        image = cv2.imread(png_file)
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
                # Calculate the total number of pixels in the subset
                total_pixels = subset.size
                # Calculate the threshold for considering an image as road
                road_pixel_threshold = int(road_threshold * total_pixels)
                # Count the number of pixels with values different than "1" in the subset
                non_1_pixels = np.count_nonzero(subset[:, :, 0] != 1)
                # Determine the category based on the percentage of non-"1" pixels
                if non_1_pixels == 0:
                    # Image does not contain any other value than "0"
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_no_road, f'{identifier}_{i}_{j}.png')
                elif non_1_pixels > road_pixel_threshold:
                    # Image contains values different than "1" in more than 20% of the pixels
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_road, f'{identifier}_{i}_{j}.png')
                else:
                    # Image contains values different than "1" in less than 20% of the pixels
                    identifier = os.path.basename(file).split(".png")[0]
                    output_path = os.path.join(output_path_do_not_use, f'{identifier}_{i}_{j}.png')
                # Save the subset in the appropriate folder
                cv2.imwrite(output_path, subset)