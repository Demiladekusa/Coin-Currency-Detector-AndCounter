import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

# Load the model
model = load_model("coin_classifier_model.keras")

# Check the model summary
model.summary()

# Perform a simple prediction test (dummy data)
import numpy as np
dummy_input = np.random.rand(1, 128, 128, 3)  # Random input data
prediction = model.predict(dummy_input)
print(prediction)
