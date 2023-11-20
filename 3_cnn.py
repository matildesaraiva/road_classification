import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Disable unnecessary logging
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

batch_size = 32
img_height = 32
img_width = 32
dataset_path = 'C:/Users/LENOVO/Desktop/thesis/data/2_dataset/raster/'

# Load the dataset with class names
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Extract class names
class_names = train_ds.class_names

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Display some example images
plt.figure(1, figsize=(10, 10))
for x_batch, y_batch in train_ds.take(1):
    batch_size = len(x_batch)
    for i in range(min(16, batch_size)):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(x_batch[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(y_batch[i, :])])
        plt.axis("off")
plt.show()

# Memory optimizations
train_ds = train_ds.cache()
val_ds = val_ds.cache()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build the model for binary classification with Batch Normalization and Dropout
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Adjust the dropout rate
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adjust the learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback for early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Adjust the patience value
    restore_best_weights=True)

model.summary()

max_epochs = 100  # Adjust the number of epochs

history = model.fit(
    train_ds,
    epochs=max_epochs,
    validation_data=val_ds,
    callbacks=[early_stopping_callback])

# Evaluate the model on the validation set
y_pred = model.predict(val_ds)
y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in val_ds], axis=0)
y_true = tf.argmax(y_true, axis=1)

num_misses = np.count_nonzero(y_true - y_pred)
num_predictions = len(y_true)
accuracy = 100.0 - num_misses * 100.0 / num_predictions
print("Val accuracy = %.02f %%" % accuracy)

# Save misclassified images into a folder
misclassified_folder = 'C:/Users/LENOVO/Desktop/thesis/data/'
os.makedirs(misclassified_folder, exist_ok=True)

for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        misclassified_image = x_batch[i].numpy().astype("uint8")
        misclassified_label = class_names[y_true[i].numpy()]
        misclassified_pred = class_names[y_pred[i].numpy()]
        misclassified_filename = f"misclassified_{i}_true_{misclassified_label}_pred_{misclassified_pred}.png"
        misclassified_path = os.path.join(misclassified_folder, misclassified_filename)
        cv2.imwrite(misclassified_path, misclassified_image)

# Plot training history and confusion matrix
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

plt.show()