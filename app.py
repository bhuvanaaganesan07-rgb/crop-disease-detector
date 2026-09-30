import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
import os
from PIL import Image

MODEL_PATH = "crop_model.keras"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    url = "https://drive.google.com/uc?id=1aTJAAkhPOalTQdqSsLiVmNRSk7-_cH8s"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Crop Disease Detector")
st.write("Upload a leaf photo to detect disease")

uploaded = st.file_uploader("Choose leaf image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Leaf")

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    disease = class_names[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)

    st.success(disease.replace("_", " ").title())
    st.write("Confidence:", confidence, "%")