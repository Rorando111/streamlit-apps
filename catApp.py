
import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model('cat_classifier.h5',compile=False)
lab = {'Abyssinian': 0, 'Bengal': 1, 'Birman': 2, 'Bombay': 3, 'British Shorthair': 4, 'Egyptian Mau': 5, 'Maine Coon': 6, 'Norwegian Forest': 7, 'Persian': 8, 'Ragdoll': 9, 'Russian Blue': 10, 'Siamese': 11, 'Sphynx': 12}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = img1.resize((224,224))
    st.image(img1,use_column_width=False)
    st.title("Cat Breed Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Cat breed is: "+result)
run()
