import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norwegian Forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)

    def get_config(self):
        config = super(CustomBatchNormalization, self).get_config()
        # Add any custom configuration here
        return config

@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the cat breed classifier model"""
    model_path = 'cat_breed_classifier.h5'
    custom_objects = {'CustomBatchNormalization': CustomBatchNormalization}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def import_and_resize_image(image_data: bytes) -> Image:
    """Import and resize the image"""
    image = Image.open(BytesIO(image_data))
    image = ImageOps.fit(image, IMAGE_SIZE, Image.LANCZOS)
    return image

def preprocess_image(image: Image) -> np.ndarray:
    """Preprocess the image for prediction"""
    img = np.asarray(image)
    img = img[np.newaxis,...]
    return img

def make_prediction(image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """Make a prediction using the model"""
    prediction = model.predict(image)
    return prediction

def display_image(image: Image) -> None:
    """Display the image using Streamlit"""
    st.image(image, use_column_width=True)

def display_prediction(prediction: np.ndarray) -> None:
    """Display the prediction result"""
    class_index = np.argmax(prediction)
    output_string = f"OUTPUT: {CLASS_NAMES[class_index]}"
    st.success(output_string)

def main() -> None:
    st.write("""
# Cat Breed Classifier
""")
    file = st.file_uploader("Choose a cat photo from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image_data = file.read()
            image = import_and_resize_image(image_data)
            display_image(image)
            preprocessed_image = preprocess_image(image)
            model = load_model()
            prediction = make_prediction(preprocessed_image, model)
            display_prediction(prediction)
        except IOError as e:
            st.error(f"Error reading image file: {e}")
        except ValueError as e:
            st.error(f"Error processing image: {e}")
if __name__ == "__main__":
    main()
