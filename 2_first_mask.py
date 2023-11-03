# Description of the notebook:
# Convert the vector's imagery into a binary mask

import cv2
import os
from PIL import Image

# Path to the folder containing the color images
input_folder_path = 'C:/Users/LENOVO/Desktop/thesis/original_vector/'
# List all the files in the folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.png'):
        # Construct the full path to the color image
        input_file_path = os.path.join(input_folder_path, filename)
        # Load the color image
        image = cv2.imread(input_file_path)
        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold the grayscale image to create a binary mask
        _, mask = cv2.threshold(gray_image, 1, 1, cv2.THRESH_BINARY)
        # Save the one-layer mask as a single-channel image with 0 and 1 values
        output_folder_path = 'C:/Users/LENOVO/Desktop/thesis/vector/'
        mask_path = os.path.join(output_folder_path, filename)
        Image.fromarray(mask).save(mask_path)