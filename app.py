import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 128
class_names = ['COVID', 'Normal', 'Viral Pneumonia']

# Title
st.title("🩻 COVID-19 Chest X-ray Classifier")
st.markdown("Upload a chest X-ray image to detect **COVID-19**, **Normal**, or **Viral Pneumonia**.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid19_cnn_model.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess image
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Show result
    st.subheader("🧠 Prediction Result")
    st.write(f"**Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
