import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load model
@st.cache(allow_output_mutation=True)
def load_model() -> tf.keras.Model:
    model = tf.keras.models.load_model('cat_classifier.h5')
    return model

model = load_model()

# UI components
st.write("""
# Cat Breed Classifier
""")

file_uploader = st.file_uploader("Upload an image of a cat to classify its breed:", type=["jpg", "png"])

def predict_breed(image: Image.Image, model: tf.keras.Model) -> str:
    size = (64, 64)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                   'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                   'Norwegian Forest', 'Persian', 'Ragdoll',
                   'Russian Blue', 'Siamese', 'Sphynx']
    return class_names[np.argmax(prediction)]

if file_uploader is not None:
    try:
        image = Image.open(file_uploader)
        st.image(image, use_column_width=True)
        prediction = predict_breed(image, model)
        st.success(f"OUTPUT: {prediction}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.text("Please upload an image file")
