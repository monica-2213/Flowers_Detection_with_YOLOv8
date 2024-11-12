import streamlit as st
import requests
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

# Function to download the model from GitHub
def download_model_from_github(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.success(f"Model downloaded successfully to {model_path}")

# URL of the model file hosted on GitHub
github_model_url = "https://github.com/monica-2213/Flowers_Detection_with_YOLOv8/raw/main/best.pt"

# Local path where the model will be stored
model_path = "/tmp/best.pt"  # or choose another directory

# Check if the model already exists
if not os.path.exists(model_path):
    # Download the model if not already downloaded
    download_model_from_github(github_model_url, model_path)
else:
    st.success("Model is already downloaded.")

# Load the YOLOv8 model
model = YOLO(model_path)

# Title for the app
st.title("YOLOv8 Flower Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file)

    # Convert the image to BGR format for OpenCV
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run inference
    results = model.predict(img_bgr)

    # Show the result on the Streamlit app
    st.image(results.render()[0], caption="Predicted Image", use_column_width=True)

    # Optionally, save the result
    output_path = "predicted_image.jpg"
    results.save(save_dir=output_path)

    # Display a success message
    st.success("Prediction complete!")
    st.write(f"Prediction results saved at: {output_path}")
