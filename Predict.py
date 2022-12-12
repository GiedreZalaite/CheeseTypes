import streamlit as st
import pathlib
from PIL import Image
from io import BytesIO, StringIO
import pickle

from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
st.empty()
st.header('Big Data assignment: Cheese Classicifcation')


model_name='FastAI.pkl'

 

model = pickle.load(open('fastAI.pkl', 'rb'))

image = st.file_uploader("Upload a picture of cheese to classify", type=["png","jpg"])
show_image = st.empty()

if not image:
    show_image.info("Choose a file to upload, only type: png, jpg")
    
else:
    image = PILImage.create((image))
    show_image.image(image)
    prediction, extradata1, extradata2 = model.predict(image)
    if prediction == 'Blue+Danish+Cheese':
        cheese_type= 'Blue Danish'
    elif prediction == 'Brie+Cheese':
        cheese_type='Brie'
    elif prediction =='Cottage+Cheese':
        cheese_type= 'Cottage'
    elif prediction == 'Feta+Cheese':
        cheese_type= 'Feta'
    elif prediction == 'Parmesan+Cheese':
        cheese_type='Parmesan'

    st.write(f'Your uploaded cheese is : {cheese_type}')
    st.write(f'Your uploaded cheese is : {cheese_type}')
    if extradata2[extradata1]<0.80:
        st.write("The uploaded image might no be a cheese as the probability is not high enough.")
    probab=tf.math.round(extradata2[extradata1]*100)   
    st.write(f"Probability: {probab}%")


