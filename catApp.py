import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

model = load_model('cat_classifier.h5', compile=False)
lab = {0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'British Shorthair', 
       5: 'Egyptian Mau', 6: 'Maine Coon', 7: 'Norwegian Forest', 8: 'Persian', 9: 'Ragdoll', 
       10: 'Russian Blue', 11: 'Siamese', 12: 'Sphynx'}

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = y_class[0]
    res = lab[y]
    return res

def run():
    st.title("Cat Breed Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Cat", type=["jpg", "png"])
    if img_file is not None:
        img1 = Image.open(img_file)
        st.image(img1, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Cat breed is: " + result)

run()
