import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from model import build_model

MODEL_PATH = '../saved_model/trained_model_final_95_percente.keras'
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    '1', '5', '10', '10 (new)', '20', '20 (new)', '50', '100', '200'
]

app = Flask(__name__)

try:
    model = build_model(num_classes=9)
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize(IMG_SIZE)

        img_array = np.array(image, dtype=np.float32) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return jsonify({
            'class': predicted_class_name,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)