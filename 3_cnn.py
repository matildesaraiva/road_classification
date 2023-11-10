# Description of the notebook:
# Running a basic CNN in an attempt to classify images
import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications import MobileNet

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt


batch_size = 32
img_height = 32
img_width = 32
dataset_path = "C:/Users/LENOVO/Desktop/thesis/raster_pieces/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',  # Updated for binary classification
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',  # Updated for binary classification
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

labels = train_ds.class_names
print(labels)

# Display some example images
plt.figure(1, figsize=(10, 10))
for x_batch, y_batch in train_ds.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(x_batch[i].numpy().astype("uint8"))
        plt.title(labels[np.argmax(y_batch[i,:])])
        plt.axis("off")
plt.show()

# Memory optimizations
train_ds = train_ds.cache()
val_ds = val_ds.cache()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Import MobileNet and make it non-trainable
MobileNetmodel = MobileNet(input_shape=(img_height, img_width, 3), include_top=False)
MobileNetmodel.summary()
MobileNetmodel.trainable = False

# Build the model for binary classification
model = tf.keras.models.Sequential([
    layers.Rescaling(1./127, offset=-1, input_shape=(img_height, img_width, 3)),
    #layers.RandomFlip("horizontal"),
    #layers.RandomRotation(0.1),
    MobileNetmodel,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

model.summary()

max_epochs = 30

# Callbacks for model checkpoint and early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="C:/LENOVO/Desktop/thesis/best_model.h5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3)

history = model.fit(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    callbacks=[model_checkpoint_callback, early_stopping_callback])

y_pred = model.predict(val_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in val_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)

num_misses = np.count_nonzero(y_true-y_pred)
num_predictions = len(y_true)
print("Val accuracy = %.02f %%" % (100.0 - num_misses * 100.0 / num_predictions))

# gerar graficos e matriz de confusao
cm = confusion_matrix(y_true, y_pred)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(history.history['accuracy']))

plt.figure(2, figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()






