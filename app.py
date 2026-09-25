import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

model = tf.keras.models.load_model("crop_model.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Crop Disease Detector")
st.write("Upload a leaf photo to detect disease")

uploaded = st.file_uploader("Choose leaf image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    st.image(img, caption="Uploaded Leaf")
    
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    
    pred = model.predict(arr)
    disease = class_names[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)
    
    st.success(disease.replace("_", " ").title())
    st.write("Confidence:", confidence, "%")