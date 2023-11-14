# Description of the notebook:
# Raster's second cut: Split each image into 32x32 pixel pieces

import cv2
import os

groundtruth_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/no_road/'
groundtruth_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/groundtruth/road/'
groundtruth_excess_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/no_road/'
groundtruth_excess_road = 'C:/Users/LENOVO/Desktop/thesis/data/3_excess/groundtruth/road/'

def split_and_save_image(input_path):
    for file in os.listdir(input_path):
        if file.endswith('.tif'):
            tif_file = os.path.join(input_path, file)
            print(tif_file)
            # Read the image using OpenCV
            image = cv2.imread(tif_file)
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
                    # Create a new PNG file for each piece
                    identifier = os.path.basename(tif_file).split(".tif")[0]
                    piece_name = f"{identifier}_{i}_{j}.png"
                    # Condition for the attribution of file to each folder
                    if os.path.exists(os.path.join(groundtruth_balanced_no_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/raster/no_road/{piece_name}'
                    elif os.path.exists(os.path.join(groundtruth_balanced_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/raster/road/{piece_name}'
                    elif os.path.exists(os.path.join(groundtruth_excess_no_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/3_excess/raster/no_road/{piece_name}'
                    elif os.path.exists(os.path.join(groundtruth_excess_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/3_excess/raster/road/{piece_name}'
                    else:
                        output_path = None
                    if output_path:
                        # Save the subset as a new image
                        cv2.imwrite(output_path, subset)

    print('Finished successfully')

if __name__ == '__main__':
    input_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/raster/'
    split_and_save_image(input_path)
