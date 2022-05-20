import streamlit as st
import tensorflow as tf
import keras
from classes import color_classes, type_classes, material_classes

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('final_model.h5')
    return model

model = load_model()
st.write("""
# Dress Features Recognition
""")

file = st.file_uploader("Please upload an image of a dress", type=['jpg','png'])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img/255.
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model=model)

    dress_color = "The dress colour is: "+color_classes[np.argmax(predictions[0])]
    dress_type = "The dress type is: " + type_classes[np.argmax(predictions[1])]
    dress_material = "The dress material is:" + material_classes[np.argmax(predictions[2])]
    st.success(dress_type)
    st.success(dress_color)
    st.success(dress_material)
