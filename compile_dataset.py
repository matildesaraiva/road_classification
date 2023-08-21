import cv2
from PIL import Image
import numpy as np
import h5py
import os
import numpy as np
from PIL import Image
import glob

image_folder = "C:/data/raster_data/"
mask_folder = "C:/data/mask_data/"

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
        if mask_img.size == (256, 256):
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

# Create TRAIN dataset
with h5py.File("C:/data/Dataset_train.h5", 'w') as hdf:
    hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)