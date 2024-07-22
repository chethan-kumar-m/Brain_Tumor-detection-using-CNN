import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

# Load the trained model and class indices
model = load_model('2_brain_tumor_detection_model.h5')

with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

class_names = list(class_indices.keys())

st.title('Brain Tumor Detection')
st.image('img1.jpeg',width=750)

st.write("""
Upload an MRI image of the brain to detect if there is a tumor.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")

def predict(image, model):
    image = image.resize((150, 150))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict(image, model)

    if prediction > 0.5:
        st.write(f"Prediction: Tumor detected with probability {prediction[0][0]:.2f}")
    else:
        st.write(f"Prediction: No tumor detected with probability {1 - prediction[0][0]:.2f}")