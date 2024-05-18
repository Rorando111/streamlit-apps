import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
    model = tf.keras.models.load_model('cat_classifier.h5', custom_objects=custom_objects)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

st.write("""
# Cat Breed Classifier
""")

file = st.file_uploader("Upload an image of a cat to classify its breed:", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    size = (224, 224)

    def import_and_predict(image_data, model):
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        img = np.asarray(image)
        img = img / 255.0
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    prediction = import_and_predict(image, model)
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                     'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                     'Norweigian forest', 'Persian', 'Ragdoll',
                     'Russian Blue', 'Siamese', 'Sphynx']

    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
