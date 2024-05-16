# catApp.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    try:
        image = Image.open(image)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def make_prediction(model, image):
    try:
        predictions = model.predict(image[None, ...])
        top_prediction = np.argmax(predictions)
        return top_prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.title("Cat Breed Classifier")
    st.write("Upload an image of a cat to classify its breed:")

    # Get the uploaded file
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])

    if uploaded_file is None:
        st.write("Please upload an image file")
        return

    # Load the model
    model = load_model('cat_classifier.h5')
    if model is None:
        return

    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    if image is None:
        return

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    top_prediction = make_prediction(model, image)
    if top_prediction is None:
        return

    # Display the result
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                   'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                   'Norweigian forest', 'Persian', 'Ragdoll',
                   'Russian Blue', 'Siamese', 'Sphynx']
    st.success(f"Predicted breed: {class_names[top_prediction]}")
