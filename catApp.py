import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('cat_breed_classifier.h5')
  return model

model=load_model()

st.write("""
# Cat Breed Classifier

Upload an image of a cat to predict its breed.
""")

file=st.file_uploader("Choose cat photo from device",type=["jpg","jpeg","png"])

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)

    # Preprocess the image
    size=(224,224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image) / 255.0
    img_reshape = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model.predict(img_reshape)

    # Get the predicted breed and probability
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                   'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                   'Norweigian forest', 'Persian', 'Ragdoll',
                   'Russian Blue', 'Siamese', 'Sphynx']
    predicted_breed = class_names[np.argmax(prediction)]
    probability = np.max(prediction)

    # Display the results
    string="OUTPUT : {} ({:.2f}%)".format(predicted_breed, probability*100)
    st.success(string)

    
