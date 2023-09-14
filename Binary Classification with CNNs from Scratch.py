import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and preprocess your image dataset.
# Make sure your dataset is organized into two folders: one for genuine images and one for counterfeit images.
# Use a data generator to load and preprocess the images.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the image dimensions and batch size
image_size = (224, 224)
batch_size = 32

# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift the width
    height_shift_range=0.2,  # Randomly shift the height
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2,  # Split data into training and validation sets
)

# Load images from the directory
train_generator = datagen.flow_from_directory(
    r"C:\Users\ALI\Desktop\All data\CNN\train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",  # Binary classification (genuine vs. counterfeit)
    subset="training",  # Specify 'training' for the training set
)

validation_generator = datagen.flow_from_directory(
    r"C:\Users\ALI\Desktop\All data\CNN\validation",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",  # Specify 'validation' for the validation set
)

# Define the CNN model
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model to a file
model.save("banknote_classification_model.h5")
print("Model saved as 'banknote_classification_model.h5'")

# You can also plot training history to visualize model performance
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
