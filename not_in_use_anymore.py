##########################################################################################
##########################################################################################
# CRIAÇÃO DE UMA IMAGEM QUE É A COLAGEM DO RASTER COM O MAPA
import numpy as np
import cv2

# This part is going to be deleted, it only has the purpose of making it easier to access if both images match
background = cv2.imread('piece_0_6.tif')
overlay = cv2.imread('C:/data/vector.tif')
# Set the alpha value of white pixels in the overlay image to zero
non_black_pixels = np.all(overlay != [0, 0, 0], axis=-1)
overlay[non_black_pixels] = [0, 0, 255]
# Merge the background and modified overlay images
combined_image = cv2.addWeighted(background, 1, overlay, 1, 0)
# Save the combined image
cv2.imwrite('C:/data/nameofthefile, combined_image)


##########################################################################################
##########################################################################################
# CONFIRMAR SE O ARRAY ESTÁ EM BINÁRIO E FOI CORRETAMENTE EXTRAÍDO
import rasterio
import numpy as np
import cv2
vec = rasterio.open("PATH/TO/FILE")
image = vec.read()
print(image.shape)
image[0][1]

##########################################################################################
##########################################################################################
# ENGROSSAR A LINHA DO VECTOR (DUPLICAR E GUARDAR UMA NOVA IMAGEM)

modified_image = np.copy(image)
for row in range(image.shape[1]):
    for col in range(1, image.shape[2] - 1):
        pixel_value = image[0, row, col]
        if pixel_value == 1:
            modified_image[0, row, col - 1] = 1
            modified_image[0, row, col + 1] = 1
modified_image[0][1]
output_path = 'C:/data/vector.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=modified_image.shape[1],
    width=modified_image.shape[2],
    count=1,
    dtype=modified_image.dtype,
    crs=rasterio.crs.CRS.from_epsg(4326),
    transform=vec.meta['transform']) as dst:
    # Write the modified_image array to the raster dataset
    dst.write(modified_image)

########################################################################################################################
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

########################################################################################################################
########################################################################################################################

# Description of the notebook:
# Raster's second cut: Split each image into 32x32 pixel pieces

from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
import os
import rasterio
from rasterio.transform import Affine
from pyproj import CRS, Transformer

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)
    # Convert the CRS
    # standard WGS84 coordinates
    new_crs = CRS.from_epsg(4326)
    # Figuring out the top left coordinates in WGS84 system
    topleft_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
    # Figuring out the bottom right coordinates in WGS84 system
    bottomright_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])

    # Read the image data
    image = dataset.read()
    # Get the dimensions of the image
    height, width = image.shape[1:]
    # Get the geospatial information
    transform_dataset = dataset.transform
    crs = dataset.crs
    # Create a list to store the coordinates and saved file paths
    piece_info = []

    # Split the image into 3x3 grid
    for i in range(96):
        for j in range(186):
            # Calculate the window bounds for each piece
            start_h = int(i * height / 96)
            end_h = int((i + 1) * height / 96)
            start_w = int(j * width / 186)
            end_w = int((j + 1) * width / 186)

            # Read the subset of the image using window
            window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
            subset = dataset.read(window=window)

            # Create a new TIFF file for each piece
            identifier = os.path.basename(tif_file).split(".tif")[0]
            piece_path = f"C:/Users/LENOVO/Desktop/thesis/raster_pieces/{identifier}_{i}_{j}.tif"
            with rasterio.open(
                piece_path,
                'w',
                driver='GTiff',
                height=subset.shape[1],
                width=subset.shape[2],
                count=subset.shape[0],
                dtype=subset.dtype,
                crs=crs,
                transform=transform_dataset * Affine.translation(start_w, start_h)
            ) as dst:
                # Write the subset to the TIFF file
                dst.write(subset)

            # Calculate the geographical coordinates of the piece
            piece_coords = transform_dataset * (start_w, start_h)
            piece_info.append((piece_coords, piece_path))

    # Print the piece coordinates and file paths
    for info in piece_info:
        print(f"Coordinates: {info[0]}")
        print(f"File path: {info[1]}")

if __name__ == '__main__':
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/to_raster/'
    for file in os.listdir(path):
        if file.endswith('.tif'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        raster_extraction(index, tif_file)

    print('Finished successfully')


########################################################################################################################
########################################################################################################################

# Description of the notebook:
# Creating train and test datasets for segmentation model

import h5py
import os
import numpy as np
from PIL import Image
import glob

image_folder = "C:/Users/LENOVO/Desktop/thesis/raster_pieces/"
mask_folder = "C:/Users/LENOVO/Desktop/thesis/vector_pieces/"

images = []
masks = []

# List all image files in the image_folder
image_files = glob.glob(os.path.join(image_folder, "*.tif"))

for image_file in image_files:
    image_name = os.path.basename(image_file).split("\\")[-1]
    mask_file = os.path.join(mask_folder, image_name)

    if os.path.exists(mask_file):
        # Process mask
        mask_img = Image.open(mask_file)
        if mask_img.size == (32, 32):
            mask_arr = np.array(mask_img)
            mask_arr = np.expand_dims(mask_arr, -1)
            masks.append(mask_arr)

            img = Image.open(image_file)
            arr = np.array(img)
            images.append(arr)
        else:
            continue

images = np.array(images)
masks = np.array(masks)
masks.shape

data_size = len(images)
indices = np.arange(data_size)
np.random.shuffle(indices)

# Split data into train and test
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(data_size * split_ratio)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

train_images = [images[i] for i in train_indices]
train_masks = [masks[i] for i in train_indices]

test_images = [images[i] for i in test_indices]
test_masks = [masks[i] for i in test_indices]

# Save the datasets
with h5py.File("C:/Users/LENOVO/Desktop/thesis/Dataset_train.h5", 'w') as hdf:
    hdf.create_dataset('images', data=train_images, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=train_masks, compression='gzip', compression_opts=9)

with h5py.File("C:/Users/LENOVO/Desktop/thesis/Dataset_test.h5", 'w') as hdf:
    hdf.create_dataset('images', data=test_images, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=test_masks, compression='gzip', compression_opts=9)

########################################################################################################################
########################################################################################################################

# Description of the notebook:
# Running a basic CNN in an attempt to identify new roads (segmentation)

from tensorflow.keras import layers, models
import h5py

def create_cnn_model(input_shape=(32, 32, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def load_datasets(train_file, test_file):
    with h5py.File(train_file, 'r') as hdf:
        train_images = hdf['images'][:]
        train_masks = hdf['masks'][:]

    with h5py.File(test_file, 'r') as hdf:
        test_images = hdf['images'][:]
        test_masks = hdf['masks'][:]

    return train_images, train_masks, test_images, test_masks

def train_model(model, train_images, train_masks, test_images, test_masks, epochs=10, batch_size=32):
    model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_masks))

if __name__ == "__main__":
    train_file = "C:/Users/LENOVO/Desktop/thesis/Dataset_train.h5"
    test_file = "C:/Users/LENOVO/Desktop/thesis/Dataset_test.h5"

    train_images, train_masks, test_images, test_masks = load_datasets(train_file, test_file)

    cnn_model = create_cnn_model()
    cnn_model.summary()

    train_model(cnn_model, train_images, train_masks, test_images, test_masks)

    # Save the model weights
    model.save_weights("C:/Users/LENOVO/Desktop/thesis/test1_weights.h5")

########################################################################################################################
########################################################################################################################

# Description of the notebook:
# Cuting the vectors into yhr shape that is needed
from rasterio.crs import CRS
import os
import rasterio
from pyproj import CRS, Transformer

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)
    image = dataset.read()

    # Define the desired dimensions
    target_width, target_height = 5952, 3040

    # Create a new directory to save the modified images
    output_dir = 'C:/Users/LENOVO/Desktop/thesis/new_vector/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")  # Debugging statement

    # Read the subset of the image (first 5952x3040 pixels from the top-left corner)
    subset = image[:, :target_height, :target_width]  # Corrected dimensions

    # Save the subset with the same filename in the output directory
    output_file = os.path.join(output_dir, os.path.basename(tif_file))
    print(f"Output File: {output_file}")  # Debugging statement
    with rasterio.open(
        output_file,
        'w',
        driver='PNG',
        width=subset.shape[2],
        height=subset.shape[1],
        count=subset.shape[0],  # Set count to the number of bands in the subset
        dtype=subset.dtype
    ) as dst:
        dst.write(subset)

if __name__ == '__main__':
    piece_info = []  # Initialize piece_info list
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/vector/'
    for file in os.listdir(path):
        if file.endswith('.png'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        print(f"Processing file: {tif_file}")  # Debugging statement
        raster_extraction(index, tif_file)

    print('Finished successfully')

#############################################################################################################
# Description of the notebook:
# Raster's second cut: Split each image into 32x32 pixel pieces

from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import Window
import os
import rasterio
from rasterio.transform import Affine
from pyproj import CRS

def raster_extraction(index, tif_file):
    print(f'{index} - {tif_file}')
    dataset = rasterio.open(tif_file)
    # Convert the CRS to standard WGS84 coordinates
    new_crs = CRS.from_epsg(4326)
    topleft_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[0]], ys=[dataset.bounds[3]])
    bottomright_coo = transform(dataset.crs, new_crs, xs=[dataset.bounds[2]], ys=[dataset.bounds[1]])
    # Read the image data
    image = dataset.read()
    # Get the dimensions of the image
    height, width = image.shape[1:]
    # Get the geospatial information
    transform_dataset = dataset.transform
    crs = dataset.crs
    # Create a list to store the coordinates and saved file paths
    piece_info = []

    # Split the image into 95x186 grid
    for i in range(95):
        for j in range(186):
            # Calculate the window bounds for each piece
            start_h = int(i * height / 95)
            end_h = int((i + 1) * height / 95)
            start_w = int(j * width / 186)
            end_w = int((j + 1) * width / 186)
            # Read the subset of the image using window
            window = rasterio.windows.Window(start_w, start_h, end_w - start_w, end_h - start_h)
            subset = dataset.read(window=window)
            # Create a new TIFF file for each piece
            identifier = os.path.basename(tif_file).split(".tif")[0]
            piece_name = f"{identifier}_{i}_{j}.png"
            # Vector folders to separate the pieces into categories (different directories)
            no_road_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/no_road/'
            road_center_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_center/'
            road_other_folder = 'C:/Users/LENOVO/Desktop/thesis/groundtruth_pieces/road_other/'
            # Condition for the attribution of file to each folder
            if os.path.exists(os.path.join(no_road_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/no_road/{piece_name}'
            elif os.path.exists(os.path.join(road_center_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_center/{piece_name}'
            elif os.path.exists(os.path.join(road_other_folder, piece_name)):
                output_path = f'C:/Users/LENOVO/Desktop/thesis/raster_pieces/road_other/{piece_name}'
            else:
                output_path = None
            if output_path:
                with rasterio.open(
                    output_path,
                    'w',
                    driver='PNG',
                    height=subset.shape[1],
                    width=subset.shape[2],
                    count=subset.shape[0],
                    dtype=subset.dtype,
                    crs=crs,
                    transform=transform_dataset * Affine.translation(start_w, start_h)
                ) as dst:
                    # Write the subset to the TIFF file
                    dst.write(subset)
            # Calculate the geographical coordinates of the piece
            piece_coords = transform_dataset * (start_w, start_h)
            piece_info.append((piece_coords, output_path))
    print(output_path)

if __name__ == '__main__':
    raster_files = []
    path = 'C:/Users/LENOVO/Desktop/thesis/raster/'
    for file in os.listdir(path):
        if file.endswith('.tif'):
            tif_file = os.path.join(path, file)
            raster_files.append(tif_file)

    for index, tif_file in enumerate(raster_files):
        raster_extraction(index, tif_file)

    print('Finished successfully')
