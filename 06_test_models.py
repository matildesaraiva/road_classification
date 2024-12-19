import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score
import pandas as pd

folder = 102

# Load Groundtruth Data
folder_groundtruth_path = 'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/groundtruth/binary/border/'
# Load Model
model_h5 = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/best_weights.h5'
# Load Raster Data
folder_raster_path = 'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/'

classification = {
    "groundtruth_piece_name" : [],
    "groundtruth" : [],
    "prediction" : []}

for file in os.listdir(folder_groundtruth_path):
    if file.endswith('.png'):
        groundtruth_file = os.path.join(folder_groundtruth_path, file)
        groundtruth = cv2.imread(groundtruth_file)
        print(groundtruth_file)
        height, width, _ = groundtruth.shape
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
                # Ensure the subset has 64x64 pixels in height and width
                subset = groundtruth[start_h:end_h, start_w:end_w, :]

                if subset.shape[0] == piece_size and subset.shape[1] == piece_size:
                    identifier = os.path.basename(groundtruth_file).split(".png")[0]
                    piece_name = f"{identifier}_{i}_{j}"

                    if np.all(subset == 0):
                        label = 0
                    else:
                        non_zero = np.count_nonzero(subset)
                        total = subset.size
                        portion = non_zero / total
                        if portion >= 0.1:
                            label = 1
                        else:
                            label = 0
                    classification['groundtruth_piece_name'].append(piece_name)
                    classification['groundtruth'].append(label)

model = load_model(model_h5)

for file in os.listdir(folder_raster_path):
    if file.endswith('.tif'):
        raster_file = os.path.join(folder_raster_path, file)
        raster = cv2.imread(raster_file)
        print(raster_file)
        height, width, _ = raster.shape
        piece_size = 64
        height_pieces = height // piece_size
        width_pieces = width // piece_size
        for i in range(height_pieces):
            for j in range(width_pieces):
                start_h = i * piece_size
                end_h = (i + 1) * piece_size
                start_w = j * piece_size
                end_w = (j + 1) * piece_size
                subset = raster[start_h:end_h, start_w:end_w, :]
                if subset.shape[0] == piece_size and subset.shape[1] == piece_size:
                    identifier = os.path.basename(raster_file).split(".tif")[0]
                    piece_name = f"{identifier}_{i}_{j}"
                    piece = cv2.cvtColor(subset, cv2.COLOR_BGR2RGB)
                    piece_array =  np.array(piece)
                    piece_tensor = tf.convert_to_tensor(piece_array, dtype=tf.float32)
                    piece_tensor = tf.expand_dims(piece_tensor, axis=0)
                    prediction = np.round(model.predict(piece_tensor))[0][0].astype(int)
                    classification['prediction'].append(prediction)
                    print(piece_name)

# Extracting ground truth and predicted values
piece_name = classification["groundtruth_piece_name"]
groundtruth = classification["groundtruth"]
prediction = classification["prediction"]

# Check the size of the arrays
size_groundtruth = len(groundtruth)
size_prediction = len(prediction)
size_piece_name = len(piece_name)

# Print the sizes
print(f"Size of groundtruth array: {size_groundtruth}")
print(f"Size of prediction array: {size_prediction}")
print(f"Size of groundtruth_piece_name array: {size_piece_name}")

# Calculate metrics
accuracy = accuracy_score(groundtruth, prediction)
precision = precision_score(groundtruth, prediction)
recall = recall_score(groundtruth, prediction)
roc_auc = roc_auc_score(groundtruth, prediction)
f1 = f1_score(groundtruth, prediction)
fbeta = fbeta_score(groundtruth, prediction, beta=2)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F-beta(2) score: {fbeta:.4f}")

# Create a pandas dataframe containing the real and the predicted classifications
df = pd.DataFrame(classification)

# Save the DataFrame to a CSV file
df.to_csv(f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/classifications.csv', index=False)
df.head()