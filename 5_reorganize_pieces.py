import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image


# Check if the original image exists
original_image_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/raster/border/'
if not os.path.isfile(original_image_path):
    print("Original image not found.")
    exit()

# Load the CSV file with classifications
classifications_df = pd.read_csv('classifications.csv')

# Create a new image with geographical references and initialize it with a neutral color (e.g., white)
original_image = cv2.imread(original_image_path)
result_image = np.ones_like(original_image) * 255  # Initialize with white color

# Assign colors based on classifications
for index, row in classifications_df.iterrows():
    file_name = row['file_name']
    i, j = map(int, file_name.split('_')[1:3])
    groundtruth = row['groundtruth']
    prediction = row['prediction']

    color = (255, 255, 255)  # Default color (white)

    # Check if the piece is saved in "excess"
    excess_piece_path = os.path.join(excess, file_name)
    if os.path.isfile(excess_piece_path):
        color = (169, 169, 169)  # Grey (excess)

    else:
        if groundtruth == 1 and prediction == 0:  # False Negative (1-0)
            color = (0, 0, 255)  # Red
        elif groundtruth == 0 and prediction == 1:  # False Positive (0-1)
            color = (0, 255, 255)  # Yellow
        elif groundtruth == 1 and prediction == 1:  # True Positive (1-1)
            color = (0, 255, 0)  # Green

    # Apply color to the corresponding region in the result image
    start_h = i * piece_size
    end_h = (i + 1) * piece_size
    start_w = j * piece_size
    end_w = (j + 1) * piece_size

    result_image[start_h:end_h, start_w:end_w, :] = color

# Save the result image
result_image_path = 'result_image.png'
cv2.imwrite(result_image_path, result_image)
