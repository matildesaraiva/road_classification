import cv2
import os
import numpy as np
import pandas as pd
import rasterio

folder = 102

df = pd.read_csv(f"C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/classifications.csv")

image = []
position = []
for item in df['groundtruth_piece_name']:
    split = item.split('_')[0]
    image.append(split)
    posi = item.split('_')[1] + '_' + item.split('_')[-1]
    position.append(posi)

df['images'] = image
df['position'] = position

input_path = 'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/'
output_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/color_coded/'

raster_list=[]

for file in os.listdir(input_path):
    if file.endswith('.tif'):
        raster = file.split('.')[0]
        image_df = df[df['images'] == raster]
        raster_list.append(raster)

        # Size of the final image
        final_width = 5952
        final_height = 3040  # Increased height to accommodate the bar

        # Size of each smaller square
        square_size = 64

        # Height of the additional bar
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
                prediction = image_df.loc[image_df['position'] == square_label]['prediction'].to_list()[0]
                groundtruth = image_df.loc[image_df['position'] == square_label]['groundtruth'].to_list()[0]

                if prediction == 1 and groundtruth == prediction:
                    color = green_color
                if prediction == 0 and groundtruth == prediction:
                    color = blue_color
                if prediction == 1 and groundtruth != prediction:
                    color = yellow_color
                if prediction == 0 and groundtruth != prediction:
                    color = red_color
                image[start_y:end_y, start_x:end_x] = color

                #label_position = (start_x + 5, start_y + 25)  # Adjust the position for better visibility
                #cv2.putText(image, square_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Add a colored bar at the bottom
        bar_color = (100, 100, 100)  # You can change the color as needed
        image[-bar_height:, :] = bar_color

        # If you want to save the image
        cv2.imwrite(f'{output_path}{raster}.png', image)

output_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/combined/'

for i in raster_list:
    background_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/{i}.tif'
    overlay_path = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/color_coded/{i}.png'

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
    out_tif = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/combined/{i}.tif'
    meta_out = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/data/1_data/raster/border/{i}.tif'
    output = f'C:/Users/LENOVO/PycharmProjects/road_segmentation/outputs/{folder}/prediction/georeferenced/{i}.tif'

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