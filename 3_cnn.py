# Description of the notebook:
# Running a basic CNN in an attempt to classify images

import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"


import tensorflow as tf
from keras import layers
from keras.applications import MobileNet
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

batch_size = 30
img_height = 32
img_width = 32
dataset_path = "C:/Users/LENOVO/Desktop/thesis/raster_pieces/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'binary',
  validation_split=0.2,
  subset="training",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'binary',
  validation_split=0.2,
  subset="validation",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

labels = train_ds.class_names
print(labels)




