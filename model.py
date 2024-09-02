# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Input
import matplotlib.pyplot as plt

# Loading the dataset
train_dir = 'Data/coins/data/train/'
validation_dir = 'Data/coins/data/validation/'
test_dir = 'Data/coins/data/test/'

# Set the image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# ImageDataGenerator for augmentation with MobileNetV2 preprocessing
train_image_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocess input for MobileNetV2
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test, just normalize using the same preprocess function
validation_image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Flow from directory
train_data_gen = train_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
)

val_data_gen = validation_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
)

test_data_gen = validation_image_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical'
)

# Debugging and Data Handling Checks
print(f"Number of training samples: {train_data_gen.samples}")
print(f"Number of validation samples: {val_data_gen.samples}")
print(f"Number of test samples: {test_data_gen.samples}")

# Define the Transfer Learning Model
# Load the pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')

# Freeze the base model to prevent training on the initial layers
base_model.trainable = False

# Add custom layers on top of the pre-trained model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Replace Flatten with GlobalAveragePooling2D for MobileNetV2
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(211, activation='softmax')  # Adjust output to 211 classes
])

# Compile the Model with a custom learning rate
optimizer = optimizers.Adam(learning_rate=0.001)  # Lowered learning rate for fine-tuning
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(
    train_data_gen,
    epochs=40,  # Start with 30 epochs, adjust if necessary
    validation_data=val_data_gen,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_data_gen)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the model
model.save("coin_classifier_model.keras")  # Save the trained model

# Plotting the training history
plt.figure(figsize=(12, 4))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
