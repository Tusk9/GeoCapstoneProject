import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import json
import boto3
import tempfile

# Load class labels
with open('class_labels.json', 'r') as file:
    class_labels = json.load(file)

# AWS S3 Bucket and Object Key
bucket_name = 'dlmodelsbucket1'
model_file_name = 'testModel.h5'

# Initialize the S3 client with credentials from Streamlit secrets
s3 = boto3.client('s3',
                  aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                  aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                  region_name='us-east-1')

# Function to load a Keras model from a byte stream
def load_model_from_s3(bucket_name, object_name):
    """Downloads a model from S3 and loads it using a temporary file."""
    # Download the model object
    model_object = s3.get_object(Bucket=bucket_name, Key=object_name)
    model_bytes = model_object['Body'].read()

    # Use a temporary file to save the model bytes
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
        tmp.write(model_bytes)
        tmp.seek(0)  # Go to the beginning of the file

        # Load the model from this temporary file
        model = load_model(tmp.name)
    return model

try:
    model = load_model_from_s3(bucket_name, model_file_name)
    st.write("Model loaded successfully from S3")
except Exception as e:
    st.write("Error loading model from S3:", e)

img_size = (224, 224)

def predict_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=img_size)
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values
    # Use the trained model to make predictions
    predictions = model.predict(img_array)
    # Decode the predictions
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_label = [k for k, v in class_labels.items() if v == predicted_class][0]
    return predicted_label, confidence

# Streamlit App
st.title('Image Classification with Streamlit')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_to_predict = image.load_img(uploaded_file, target_size=img_size)
    st.image(image_to_predict, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    predicted_label, confidence = predict_image(uploaded_file)

    st.write("Predicted Class:", predicted_label)
    st.write("Confidence:", f"{confidence * 100:.2f}%")
