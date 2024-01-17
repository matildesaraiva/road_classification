import time
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import cv2

def start_timer():
    return time.time()

def end_timer(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# start counting the time
start_time = start_timer()

# Disable unnecessary logging
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

batch_size = 32
img_height = 64
img_width = 64
dataset_path = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium/raster/no_border/'
test_dataset_path = 'C:/Users/LENOVO/Desktop/thesis/data/2_datasets/medium/raster/border/'

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

# Memory optimizations
train_ds = train_ds.cache()
val_ds = val_ds.cache()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Build the model for binary classification with Batch Normalization and Dropout
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3), padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(1, activation="sigmoid")
])

# Ajustar o learning rate para um maior (0.001)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback for early stopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)

# Callback to save the model weights
model_checkpoint_callback = ModelCheckpoint(
    filepath='../../../best_weights.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

# Check if a weights file exists
if os.path.exists('../../../best_weights.h5'):
    model.load_weights('best_weights.h5')
    print("Loaded weights from an existing file.")

model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)

# Convert train_ds to a NumPy array
train_images = []
train_labels = []

for images, labels in train_ds:
    train_images.append(images.numpy())
    train_labels.append(labels.numpy())

train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Fit the model with data augmentation
max_epochs = 50
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=max_epochs,
    validation_data=val_ds,
    #callbacks=[early_stopping_callback, model_checkpoint_callback]
    callbacks=[model_checkpoint_callback]
)

# Save the final weights
model.save_weights('final_weights.h5')

# Evaluate the model on the validation set
y_pred_probs = model.predict(val_ds)
y_pred = np.round(y_pred_probs).flatten()

y_true = tf.concat([y for x, y in val_ds], axis=0).numpy()

# Calculate precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1_score_val = f1_score(y_true, y_pred)

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-Score: {:.4}".format(f1_score_val))

# Save misclassified images into a folder
misclassified_folder = 'C:/Users/LENOVO/Desktop/thesis/data/misclassified/'
os.makedirs(misclassified_folder, exist_ok=True)

for (x_batch, y_true_batch) in val_ds:
    y_pred_batch = model.predict(x_batch)
    y_pred_batch = np.round(y_pred_batch).flatten()

    for i in range(len(y_true_batch)):
        if y_true_batch[i].numpy() != y_pred_batch[i]:
            misclassified_image = x_batch[i].numpy().astype("uint8")
            misclassified_label = class_names[int(y_true_batch[i].numpy())]
            misclassified_pred = class_names[int(y_pred_batch[i])]
            misclassified_filename = f"misclassified_{i}_true_{misclassified_label}_pred_{misclassified_pred}.png"
            misclassified_path = os.path.join(misclassified_folder, misclassified_filename)
            cv2.imwrite(misclassified_path, misclassified_image)

# Test the model on the test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dataset_path,
    labels='inferred',
    label_mode='binary',
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = test_ds.cache()
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predictions on the test set
y_test_pred_probs = model.predict(test_ds)
y_test_pred = np.round(y_test_pred_probs).flatten()

y_test_true = tf.concat([y for x, y in test_ds], axis=0).numpy()

# Calculate precision, recall, and F1-Score for the test set
test_precision = precision_score(y_test_true, y_test_pred)
test_recall = recall_score(y_test_true, y_test_pred)
test_f1_score = f1_score(y_test_true, y_test_pred)

print("Test Precision: {:.4f}".format(test_precision))
print("Test Recall: {:.4f}".format(test_recall))
print("Test F1-Score: {:.4f}".format(test_f1_score))

# Save misclassified images from the test set into a folder
test_misclassified_folder = 'C:/Users/LENOVO/Desktop/thesis/data/test_misclassified/'
os.makedirs(test_misclassified_folder, exist_ok=True)

for (x_batch_test, y_true_batch_test) in test_ds:
    y_pred_batch_test = model.predict(x_batch_test)
    y_pred_batch_test = np.round(y_pred_batch_test).flatten()

    for i in range(len(y_true_batch_test)):
        if y_true_batch_test[i].numpy() != y_pred_batch_test[i]:
            misclassified_image_test = x_batch_test[i].numpy().astype("uint8")
            misclassified_label_test = class_names[int(y_true_batch_test[i].numpy())]
            misclassified_pred_test = class_names[int(y_pred_batch_test[i])]
            misclassified_filename_test = f"test_misclassified_{i}_true_{misclassified_label_test}_pred_{misclassified_pred_test}.png"
            misclassified_path_test = os.path.join(test_misclassified_folder, misclassified_filename_test)
            cv2.imwrite(misclassified_path_test, misclassified_image_test)

#training history
# Plot training and test metrics
plt.figure(figsize=(15, 6))

# Plot Training and Test Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Test Accuracy')

plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

# Plot Training and Test Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='green')
plt.axhline(y=test_loss, color='purple', linestyle='--', label='Test Loss')

plt.legend(loc='upper right')
plt.title('Training and Test Loss')

plt.show()

elapsed_time = end_timer(start_time)
print(f"Elapsed time: {elapsed_time} seconds")
