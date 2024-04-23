import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Load your model
model = tf.keras.models.load_model('model_weights.h5')

# Function to make predictions
def predict(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    predictions = model.predict(img_array)

    class_labels = {'biodegradable': 0, 'metal': 1, 'paper(recycle)': 2, 'plastic bag': 3, 'plastic bottle': 4}
    carbon_footprints = {'biodegradable': 7.76, 'metal': 0.096, 'paper(recycle)': 0.046, 'plastic bag': 1.58, 'plastic bottle': 0.82}
    threshold = 0.7

    results = []
    for class_label, probability in zip(class_labels.keys(), predictions[0]):
        if probability > threshold:
            carbon_footprint = carbon_footprints.get(class_label, 0)
            results.append(f"{class_label}: {probability:.2f} (Carbon Footprint: {carbon_footprint})")
    return results

# Streamlit app
st.title('GreenerVisions InceptionV3')
st.write('Upload an image to classify and get its carbon footprint.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    predictions = predict(uploaded_file)
    for result in predictions:
        st.write(result)
