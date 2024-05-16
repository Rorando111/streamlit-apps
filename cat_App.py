# catApp.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def make_prediction(model, image):
    try:
        predictions = model.predict(image[None, ...])
        top_prediction = np.argmax(predictions)
        return top_prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def main():
    print("Cat Breed Classifier")
    print("Upload an image of a cat to classify its breed:")

    # Get the uploaded file
    uploaded_file = input("Enter the file path: ")

    # Load the model
    model = load_model('cat_classifier.h5')
    if model is None:
        return

    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)
    if image is None:
        return

    # Display the uploaded image
    plt.imshow(image)
    plt.title("Uploaded Image")
    plt.show()

    # Make predictions
    top_prediction = make_prediction(model, image)
    if top_prediction is None:
        return

    # Display the result
    class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
                   'British Shorthair', 'Egyptian Mau', 'Maine Coon',
                   'Norweigian forest', 'Persian', 'Ragdoll',
                   'Russian Blue', 'Siamese', 'Sphynx']
    print(f"Predicted breed: {class_names[top_prediction]}")

if __name__ == "__main__":
    main()