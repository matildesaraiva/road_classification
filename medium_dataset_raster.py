import cv2
import os

groundtruth_balanced_no_road = 'C:/Users/LENOVO/Desktop/thesis/data/medium_dataset/groundtruth/no_border/no_road/'
groundtruth_balanced_road = 'C:/Users/LENOVO/Desktop/thesis/data/medium_dataset/groundtruth/no_border/road/'

def split_and_save_image(input_path):
    for file in os.listdir(input_path):
        if file.endswith('.tif'):
            tif_file = os.path.join(input_path, file)
            print(tif_file)
            # Read the image using OpenCV
            image = cv2.imread(tif_file)

            height, width, _ = image.shape
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

                    # Ensure the subset has 96x96 pixels in height and width
                    subset = image[start_h:end_h, start_w:end_w, :]

                    # Create a new PNG file for each piece
                    identifier = os.path.basename(tif_file).split(".tif")[0]
                    piece_name = f"{identifier}_{i}_{j}.png"
                    # Condition for the attribution of file to each folder
                    if os.path.exists(os.path.join(groundtruth_balanced_no_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/medium_dataset/raster/no_border/no_road/{piece_name}'
                    elif os.path.exists(os.path.join(groundtruth_balanced_road, piece_name)):
                        output_path = f'C:/Users/LENOVO/Desktop/thesis/data/medium_dataset/raster/no_border/road/{piece_name}'
                    else:
                        output_path = None
                    if output_path:
                        # Save the subset as a new image
                        cv2.imwrite(output_path, subset)

    print('Finished successfully')

if __name__ == '__main__':
    input_path = 'C:/Users/LENOVO/Desktop/thesis/data/1_data/raster/no_border/'
    split_and_save_image(input_path)