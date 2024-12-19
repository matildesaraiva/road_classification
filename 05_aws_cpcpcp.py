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

start_time = start_timer()

# Disable unnecessary logging
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

batch_size = 32
img_height = 64
img_width = 64
dataset_path = "data"

# Load the dataset with class names
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=1238,
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
    seed=1238,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Memory optimizations
train_ds = train_ds.cache()
val_ds = val_ds.cache()
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model with one dense layer in the last layer
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
    layers.Dense(1, activation="sigmoid")
])

# Model with two dense layers in the last layer
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

#ajustar o learning rate para um maior (0.001)
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
    filepath='112_best_weights.h5',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1)

# Check if a weights file exists
if os.path.exists('112_best_weights.h5'):
    model.load_weights('112_best_weights.h5')
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
    # without early stoppinh
    callbacks=[model_checkpoint_callback]
)

# Save the final weights
model.save_weights('112_final_weights.h5')

# Evaluate the model on the validation set
y_pred_probs = model.predict(val_ds)
y_pred = np.round(y_pred_probs).flatten()

y_true = tf.concat([y for x, y in val_ds], axis=0).numpy()

# Calculate precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Save misclassified images into a folder
misclassified_folder = 'misclassified/'
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

elapsed_time = end_timer(start_time)
print(f"Elapsed time: {elapsed_time} seconds")