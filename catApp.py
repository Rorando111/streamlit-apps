import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cat_breed_classifier.h5')
    return model

model = load_model()

st.write("""
# Cat Breed Detection System
""")

file = st.file_uploader("Choose a cat photo from your computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)  # Update the size to match the input shape of your model
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image as your model was trained with normalized images
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norwegian Forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']
  # Replace with actual cat breed names
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
