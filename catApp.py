import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('cat_classifier.hdf5')

# Create a Streamlit app
st.title("Image Classification App")

# Create a file uploader
uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Create a button to classify the image
if st.button("Classify"):
    # Load the uploaded image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    # Make predictions
    predictions = model.predict(image[np.newaxis,...])

    # Get the class label with the highest probability
    class_label = np.argmax(predictions)

    # Display the result
    st.write(f"Class label: {class_label}")
    st.write(f"Confidence: {predictions[0, class_label]:.2f}")
