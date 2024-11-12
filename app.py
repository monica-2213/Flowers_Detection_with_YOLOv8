import streamlit as st
import os
import requests
from PIL import Image
import numpy as np
import cv2

# Check if ultralytics is installed, and install it if not
try:
    from ultralytics import YOLO
    st.success("YOLOv8 model is ready!")
except ImportError:
    st.error("Failed to import YOLOv8. Installing...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Download YOLOv8 model from GitHub if not present
model_path = "/tmp/best.pt"  # Temporary path in the cloud environment
github_model_url = "https://github.com/monica-2213/Flowers_Detection_with_YOLOv8/raw/main/best.pt"

# If the model doesn't exist, download it
if not os.path.exists(model_path):
    st.write("Downloading model...")
    response = requests.get(github_model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.success("Model downloaded!")

# Load the YOLOv8 model
model = YOLO(model_path)

# Streamlit interface
st.title("Flower Detection with YOLOv8")
st.write("Upload an image of a flower, and YOLOv8 will predict the flower type.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to OpenCV format (for YOLOv8)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference on the image
    st.write("Running inference on the image...")
    results = model(img_bgr)

    # Print the structure of the results object to understand its attributes
    st.write("Results Object Structure:")
    st.write(results)

    # Safely access the prediction data based on the result structure
    try:
        st.write("Predictions:")
        for result in results.pandas().xywh[0].itertuples():  # Use pandas result
            class_name = result.name  # Access the name column from the pandas dataframe
            confidence = result.confidence  # Confidence score

            st.write(f"{class_name} - Confidence: {confidence:.2f}")

        # Display the image with bounding boxes
        annotated_image = results.plot()  # The image with detections
        st.image(annotated_image, caption="Predicted Image", use_column_width=True)

        # Optionally, you can save the predictions
        # Save the image to the server (optional)
        output_image_path = "/tmp/predicted_image.jpg"
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, annotated_image)
        st.download_button("Download Predicted Image", data=open(output_image_path, "rb"), file_name="predicted_image.jpg")

    except Exception as e:
        st.error(f"An error occurred while processing predictions: {e}")
