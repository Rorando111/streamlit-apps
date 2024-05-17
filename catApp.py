import streamlit as st
import tensorflow as tf
import streamlit as st

class_names = ['Abyssinian', 'Bengal', 'Birman', 'Bombay',
               'British Shorthair', 'Egyptian Mau', 'Maine Coon',
               'Norweigian forest', 'Persian', 'Ragdoll',
               'Russian Blue', 'Siamese', 'Sphynx']

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./cat_breeds_classifier.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Cat Breed Classification
         """
         )

file = st.file_uploader("Please upload cat image from device", type=["jpg", "jpeg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

