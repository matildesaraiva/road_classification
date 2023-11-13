import os
import cv2
import numpy as np
import random

# Load the image
image_path = "C:/Users/LENOVO/Desktop/thesis/03_balanced_groundtruth/road/0_22_167.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Flatten the 2D array into a 1D array
flat_image = image.flatten()

# Count the occurrences of each unique value
unique_values, counts = np.unique(flat_image, return_counts=True)

# Print the counts of each unique value
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")
