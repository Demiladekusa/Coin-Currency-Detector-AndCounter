from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import json

app = Flask(__name__)

# Load the coin names from the JSON file
json_path = os.path.join('Data', 'cat_to_name.json')  # Adjusted to point to the correct path
with open(json_path, 'r') as f:
    coin_info = json.load(f)

# Load the pre-trained model
model = load_model("coin_classifier_model.keras")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Load and preprocess the image for the model
        img = load_img(filepath, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image using the same method as in training

        # Make prediction using the loaded model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        # Get the coin name using the predicted class index
        coin_name = coin_info.get(str(predicted_class[0]), "Unknown coin")

        return render_template('result.html', coin=coin_name)

if __name__ == '__main__':
    app.run(debug=True)
