import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import keras
import gdown

def predict(img_path, model, class_names):
  #Input image
  img = tf.image.resize(img_path, (224, 224))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array,0)
  #Predict
  predictions = model.predict(img_array)
  predicted_class = class_names[np.argmax(predictions[0])]
  accuracy = round(100 * np.max(predictions[0]),10)
  return img, predicted_class, accuracy

st.title('Breast Cancer Classification App by :blue[Histopathological Images]')
note1 = '''Authors: Hao Mai Xuan, Duong Cao Thi Thuy, Yankuba B. Manga\n
Product from master thesis at The Master Program in Smart Healthcare Management of National Taipei University (Taiwan).
The product uses open data (BreakHis Dataset), so the classification results still need further consultation from specialists before using.\n
The aim of this application is to diagnose between benign and malignant from histopathological images.'''
st.markdown(note1)
st.image("medical-banner-with-doctor-wearing-goggles.jpg")
st.markdown("_:red[Note: Please wait a moment when changing the options below (pay attention to the notification in the top right corner)!]_")

st.subheader(":blue[Choose] Classification Model",divider="gray")
note2 = '''The models available in this software are: mixed, 40x, 100x, 200x and 400x.
The model you want to use depends on the magnification of the image you want to classify.
'''
#For example, for an image with a magnification of 40x, choose the 40x model.
#If you are not sure about the magnification, you can choose the mixed model.
st.markdown(note2)
selected_model = st.selectbox("Which model do you want to use?",('Mixed', '40x', '100x', '200x', '400x'))
#Load model
@st.cache_resource
def load_model(selected_model):
  gdown.download("https://drive.google.com/file/d/1-9hFzDES7FgSBQcBKprHEUvLTQwAuEzn/view?usp=sharing",output="mdlmixed.keras", fuzzy=True)
  gdown.download("https://drive.google.com/file/d/1-FO8EdMgAU62w4lTEaO0ebuLhaoTERZH/view?usp=sharing",output="mdl40x.keras", fuzzy=True)
  gdown.download("https://drive.google.com/file/d/1-IarjHPol74vqbqu61WWhAPBzKzsKL2i/view?usp=sharing",output="mdl100x.keras", fuzzy=True)
  gdown.download("https://drive.google.com/file/d/1-GWtcdxRbOINCNusMqrreh3fAx_CXK8B/view?usp=sharing",output="mdl200x.keras", fuzzy=True)  
  gdown.download("https://drive.google.com/file/d/1--6MCj9fwXhs_m355yAiBF5dhDc-BHPh/view?usp=sharing",output="mdl400x.keras", fuzzy=True)
  if selected_model == 'Mixed':
    mdlmixed = tf.keras.models.load_model("mdlmixed.keras")
    return mdlmixed
  elif selected_model == '40x':
    mdl40x = tf.keras.models.load_model("mdl40x.keras")
    return mdl40x
  elif selected_model == '100x':
    mdl100x = tf.keras.models.load_model("mdl100x.keras")
    return mdl100x
  elif selected_model == '200x':
    mdl200x = tf.keras.models.load_model("mdl200x.keras")
    return mdl200x
  elif selected_model == '400x':
    mdl400x = tf.keras.models.load_model("mdl200x.keras")
    return mdl400x
model = load_model(selected_model)

st.write(f"You selected: **:red[{selected_model}]** model")
if selected_model == 'Mixed':
  st.write("This model is applied in cases where the magnification level of the image to be classified cannot be determined. Therefore, it not only helps classify the two classes but also identifies the magnification level of the image.")
elif selected_model == '40x':
  st.write("This model is applied in cases where the magnification level of the image to be classified is 40x.")
elif selected_model == '100x':
  st.write("This model is applied in cases where the magnification level of the image to be classified is 100x.")
elif selected_model == '200x':
  st.write("This model is applied in cases where the magnification level of the image to be classified is 200x.")
elif selected_model == '400x':
  st.write("This model is applied in cases where the magnification level of the image to be classified is 400x.")
if selected_model == 'Mixed':
  class_names = ['Benign - 40x','Malignant - 40x','Benign - 100x','Malignant - 100x','Benign - 200x','Malignant - 200x','Benign - 400x','Malignant - 400x']
else:
  class_names = ['Benign','Malignant']

st.subheader(":blue[Upload] Histopathological Image",divider="gray")
st.markdown("Please select an image that has the same magnification as the selected model!")
with st.form("my-form", clear_on_submit=True):
  uploaded_files = st.file_uploader("Upload Image Here!",type=["jpg", "jpeg", "png"], accept_multiple_files=True)
  submitted = st.form_submit_button("**:red[CLASSIFY!]**")
  if submitted and uploaded_files is not None:
    st.write("Uploaded and classified successfully!")
    for uploaded_file in uploaded_files:
      img = Image.open(uploaded_file)
      name = uploaded_file.name
      st.image(img)
      img, predicted_class, accuracy = predict(img,model,class_names)
      st.write(f"Image: {name} - Classification Result: **:red[{predicted_class}] ({accuracy}%)**")
