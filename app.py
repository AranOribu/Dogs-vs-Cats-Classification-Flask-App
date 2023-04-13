from flask import Flask, render_template, request, redirect, url_for
import os
import base64
from PIL import Image
from io import BytesIO
import base64
import io
import numpy as np
import tensorflow as tf
from flask import jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the model
model = load_model('model/model.h5')

img_width, img_height = 128, 128

def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img



# Predict the class of an image and print the probabilities
def predict_image(image, model):
    img = preprocess_image(image, (img_width, img_height))
    img = np.reshape(img, (1, img_width, img_height, 3))  # Reshape to (1, 128, 128, 3)

    with tf.device('/CPU:0'):
        prediction = model.predict(img)[0][0]

    # Calculate the percentage of chance for both cat and dog predictions
    dog_probability = float(prediction * 100)
    cat_probability = float((1 - prediction) * 100)

    return cat_probability, dog_probability


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return redirect(request.url)

        images = []
        for file in request.files.getlist('files[]'):
            img_str = base64.b64encode(file.read()).decode('utf-8')
            img = Image.open(BytesIO(base64.b64decode(img_str)))
            images.append((img_str, img))

        predictions = [(img_str, predict_image(img, model)) for img_str, img in images]
        return render_template('index.html', predictions=predictions)

    return render_template('index.html')

#Postman
@app.route('/predict-api', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400

    cat_probability, dog_probability = predict_image(img, model)

    response = {
        "cat probability": cat_probability,
        "dog probability": dog_probability
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
