import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="bhuvanaganesan/crop-disease-model",
        filename="crop_model.keras"
    )
    return tf.keras.models.load_model(path)

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("Crop Disease Detector")
st.write("Upload a leaf photo to detect disease")

uploaded = st.file_uploader("Choose leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Leaf")

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    disease = class_names[np.argmax(pred)]
    confidence = round(float(np.max(pred)) * 100, 1)

    st.success(disease.replace("_", " ").title())
    st.write("Confidence:", confidence, "%")