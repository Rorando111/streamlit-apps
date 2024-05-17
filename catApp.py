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
    try:
        model_path = 'cat_breed_classifier.h5'
        model = tf.keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found")
        return None
    except tf.errors.OpError as e:
        st.error(f"Error loading model: {e}")
        return None

def import_and_resize_image(image_data_bytes: bytes) -> Image:
    image = Image.open(BytesIO(image_data_bytes))
    image = ImageOps.fit(image, IMAGE_SIZE, Image.LANCZOS)
    return image

def preprocess_image(image: Image) -> np.ndarray:
    img = np.asarray(image)
    img = img[np.newaxis,...]
    return img

def load_and_process_image(file: st.UploadedFile) -> np.ndarray:
    image_data_bytes = file.read()
    image = import_and_resize_image(image_data_bytes)
    preprocessed_image = preprocess_image(image)
    return preprocessed_image

def make_prediction(preprocessed_image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    prediction = model.predict(preprocessed_image)
    return prediction

def main():
    st.write("""
# Cat Breed Classifier
""")
    file = st.file_uploader("Choose a cat photo from your computer", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            preprocessed_image = load_and_process_image(file)
            model = load_model()
            if model is None:
                st.error("Error loading model")
                return
            prediction = make_prediction(preprocessed_image, model)
            class_index = np.argmax(prediction)
            output_string = f"OUTPUT: {CLASS_NAMES[class_index]}"
            st.success(output_string)
        except IOError as e:
            st.error(f"Error reading image file: {e}")
        except tf.errors.OpError as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
