from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

model = load_model("parkinson_disease_detection.h5")
labels = ['Healthy', 'Parkinson']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    try:
        temp_path = "temp_input.png"
        image_file.save(temp_path)

        image = preprocess_image(temp_path)
        prediction = model.predict(image)[0]
        os.remove(temp_path)

        prob_healthy = float(prediction[0])
        prob_parkinson = float(prediction[1])
        predicted_class = np.argmax(prediction)

        return jsonify({
            'probabilidad_healthy': round(prob_healthy * 100, 2),
            'probabilidad_parkinson': round(prob_parkinson * 100, 2),
            'diagnostico': labels[predicted_class]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "API para predicción de Parkinson funcionando."

if __name__ == '__main__':
    app.run(debug=True)