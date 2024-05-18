import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

# Custom object configuration for deserialization
custom_objects = {'BatchNormalization': BatchNormalization}

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cat_classifier.hdf5', custom_objects=custom_objects)
    return model

model = load_model()
st.write("""
# Cat Breed Classifier
""")

file = st.file_uploader("Upload an image of a cat to classify its breed:", type=["jpg", "jpeg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Preprocess the image
    img_reshape = import_and_predict(image, model)

    # Get the class label with the highest probability
    class_label = np.argmax(img_reshape)

    # Display the result
    st.write(f"Class label: {class_label}")
    st.write(f"Confidence: {img_reshape[0, class_label]:.2f}")
