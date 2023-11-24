# Description of the notebook:
# Convert the groudtruth images into a one-layer mask as a single-channel image with 0 and 1 values

import cv2
import os
from PIL import Image

input_folder_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/groundtruth/original/no_border'
output_folder_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/groundtruth/binary/no_border'

for filename in os.listdir(input_folder_path):
    if filename.endswith('.png'):
        input_file_path = os.path.join(input_folder_path, filename)
        image = cv2.imread(input_file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the color image to grayscale
        _, mask = cv2.threshold(gray_image, 1, 1, cv2.THRESH_BINARY) # Threshold the grayscale image to create a binary mask
        mask_path = os.path.join(output_folder_path, filename)
        Image.fromarray(mask).save(mask_path)