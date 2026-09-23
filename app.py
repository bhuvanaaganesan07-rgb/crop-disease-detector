from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

model = tf.keras.models.load_model('crop_model.keras')

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No file selected", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    disease = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 1)
    
    disease_clean = disease.replace('_', ' ').title()
    
    return render_template('result.html',
        disease=disease_clean,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)