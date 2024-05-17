import streamlit as st
import tensorflow as tf
import os
import numpy as np

@st.cache_resource
def load_model():
    model_path = 'cat_classifier.h5'
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.write("""
# Cat Breed Classifier
""")

file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if file is not None:
    image = tf.io.read_file(file)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 
                   'Egyptian Mau', 'Maine Coon', 'Norwegian Forest', 'Persian', 
                   'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Predicted class: {predicted_class}")
