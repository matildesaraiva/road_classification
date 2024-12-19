from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score
import pandas as pd
import cv2
import os
import numpy as np
import pandas as pd
import rasterio

folder = 112

df = pd.read_csv(f"C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/classifications

df.head()

raster = []
x_position = []
y_position = []

for i in df['groundtruth_piece_name']:
    img = i.split('_')[0]
y = i.split('_')[-1]
x = i.split('_')[1]

raster.append(img)
x_position.append(x)
y_position.append(y)

df['image'] = raster
df['row'] = x_position
df['column'] = y_position

df['post_processing'] = df['prediction']

for i, row in df.iterrows():
    img = row['image']
row_value = int(row['row'])
column_value = int(row['column'])
pred = row['prediction']
print(f'{img}_{row_value}_{column_value}___{pred}')

    if pred == 1:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Center
        if (up_pred == 0 and up_left_pred == 0 and up_right_pred == 0 and down_pred == 0 and down_left_pred == 0 and down_right_pred == 0 and left_pred == 0 and right_pred == 0):
        df.loc[i, 'post_processing'] = 0

    if pred == 1 and column_value == 0:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Center-Left
        if (up_pred == 0 and up_right_pred == 0 and down_pred == 0 and down_right_pred == 0 and right_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and column_value == 92:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Center-Right
        if (up_pred == 0 and up_left_pred == 0 and down_pred == 0 and down_left_pred == 0 and left_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and column_value == 0 and row_value == 46:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Bottom-Left
        if (up_pred == 0 and right_pred == 0 and up_right_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and row_value == 46:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Bottom-Center
        if (up_pred == 0 and up_left_pred == 0 and up_right_pred == 0 and left_pred == 0 and right_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and row_value == 46 and column_value == 92:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Bottom-Right
        if (up_pred == 0 and up_left_pred == 0 and left_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and row_value == 0 and column_value == 0:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Top-Left
        if (down_pred == 0 and down_right_pred == 0 and right_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and row_value == 0:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Top-Center
        if (down_pred == 0 and down_left_pred == 0 and down_right_pred == 0 and left_pred == 0 and right_pred == 0):
            df.loc[i, 'post_processing'] = 0

    if pred == 1 and row_value == 0 and column_value == 92:
        up = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        up_left = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        up_right = df[(df['row'] == str(int(row_value) - 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        down = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value))) & (df['image'] == img)]
        down_left = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        down_right = df[(df['row'] == str(int(row_value) + 1)) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        left = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) - 1)) & (df['image'] == img)]
        right = df[(df['row'] == str(int(row_value))) & (df['column'] == str(int(column_value) + 1)) & (df['image'] == img)]
        # Initialize variables to store neighboring cell values
        up_pred, up_left_pred, up_right_pred, down_pred, down_left_pred, down_right_pred, left_pred, right_pred = None, None, None, None, None, None, None, None
        # Check if 'up' DataFrame is not empty
        if not up.empty:
            up_pred = up.iloc[0]['prediction']
        if not up_left.empty:
            up_left_pred = up_left.iloc[0]['prediction']
        if not up_right.empty:
            up_right_pred = up_right.iloc[0]['prediction']
        if not down.empty:
            down_pred = down.iloc[0]['prediction']
        if not down_left.empty:
            down_left_pred = down_left.iloc[0]['prediction']
        if not down_right.empty:
            down_right_pred = down_right.iloc[0]['prediction']
        if not left.empty:
            left_pred = left.iloc[0]['prediction']
        if not right.empty:
            right_pred = right.iloc[0]['prediction']
        # Top-Right
        if (down_pred == 0 and down_left_pred == 0 and left_pred == 0):
            df.loc[i, 'post_processing'] = 0

# Save the DataFrame to a CSV file
df.to_csv(f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/post_processing.csv', index=False)
df.head()

# Extracting ground truth and predicted values
groundtruth = df['groundtruth']
prediction = df['prediction']
post_processing = df['post_processing']

print(f'Metrics using predicted values:')
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
print()
print(f'Metrics using post-processed values:')
# Calculate metrics
accuracy = accuracy_score(groundtruth, post_processing)
precision = precision_score(groundtruth, post_processing)
recall = recall_score(groundtruth, post_processing)
roc_auc = roc_auc_score(groundtruth, post_processing)
f1 = f1_score(groundtruth, post_processing)
fbeta = fbeta_score(groundtruth, post_processing, beta=2)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"F-beta(2) score: {fbeta:.4f}")

position = []
for item in df['groundtruth_piece_name']:
    posi = item.split('_')[1] + '_' + item.split('_')[-1]
position.append(posi)

df['position'] = position
df.head()

input_path = 'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/'
output_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/color_coded/'

raster_list=[]

for file in os.listdir(input_path):
    if file.endswith('.tif'):
        raster = file.split('.')[0]
        image_df = df[df['image'] == int(raster)]
        raster_list.append(raster)

        final_width = 5952
        final_height = 3040

        square_size = 64
        bar_height = 32

        # Calculate the number of squares in both dimensions
        num_squares_x = final_width // square_size
        num_squares_y = (final_height - bar_height) // square_size

        # Define colors for each case
        blue_color = (250, 0, 0)  # Blue - True Negative
        red_color = (0, 0, 255)  # Red - False Negative
        yellow_color = (0, 255, 255)  # Yellow - False Positive
        green_color = (0, 255, 0)  # Green - True Positive

        # Create a blank white image
        image = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255

        # Fill the image with smaller squares and label them
        for y in range(num_squares_y):
            for x in range(num_squares_x):
                start_x = x * square_size
                start_y = y * square_size
                end_x = start_x + square_size
                end_y = start_y + square_size

                # Label each square with its indices
                square_label = f"{y}_{x}"
                post_processing = image_df.loc[image_df['position'] == square_label]['post_processing'].to_list()[0]
                groundtruth = image_df.loc[image_df['position'] == square_label]['groundtruth'].to_list()[0]
                if post_processing == 1 and groundtruth == post_processing:
                    color = green_color
                if post_processing == 0 and groundtruth == post_processing:
                    color = blue_color
                if post_processing == 1 and groundtruth != post_processing:
                    color = yellow_color
                if post_processing == 0 and groundtruth != post_processing:
                    color = red_color
                image[start_y:end_y, start_x:end_x] = color

        # Add a colored bar at the bottom
        bar_color = (100, 100, 100)  # You can change the color as needed
        image[-bar_height:, :] = bar_color

        # If you want to save the image
        cv2.imwrite(f'{output_path}{raster}.png', image)

output_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/combined/'

for i in raster_list:
    background_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/{i}.tif'
overlay_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/color_coded/{i}.png'

# This part is going to be deleted, it only has the purpose of making it easier to access if both images match
background = cv2.imread(background_path)
overlay = cv2.imread(overlay_path)
# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [100, 50, 50]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 0.75, overlay, 0.4, 0)
# Save the combined image
cv2.imwrite(f'{output_path}{i}.tif', combined_image)

for i in raster_list:
    out_tif = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/combined/{i}.tif'
    meta_out = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/{i}.tif'
    output = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/post_processing/georeferenced/{i}.tif'

    # Open the source image
    with rasterio.open(out_tif) as src:
        # Read image data
        data = src.read()
        # Get profile (metadata) from the source image
        profile = src.profile

    # Open the image containing the desired CRS and transform
    with rasterio.open(meta_out) as src_meta:
        # Get CRS and transform from the metadata
        crs = src_meta.crs
        transform = src_meta.transform

    # Update the CRS and transform in the profile
    profile.update(crs=crs, transform=transform)

    # Create a new destination image with the same profile as meta_out
    with rasterio.open(output, "w", **profile) as dst:
        # Write the image data to the destination image
        for band in range(1, profile['count'] + 1):
            dst.write(data[band - 1], band)

