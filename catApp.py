import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import numpy as np

class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norweigian forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('cat_breed_classifier.h5')
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Cat Breed Classification
         """
         )

file = st.file_uploader("Please upload cat image from device", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    predictions = model.predict(img_reshape)
    score = tf.nn.softmax(predictions[0])
    return predictions, score

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions, score = import_and_predict(image, model)
    st.write("Predictions:", predictions)
    st.write("Scores:", score)
    st.success(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
