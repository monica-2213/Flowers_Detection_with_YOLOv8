import streamlit as st
import os
import requests
from PIL import Image
import numpy as np
import cv2

# Set Streamlit page configuration for better UI
st.set_page_config(page_title="Flower Detection with YOLOv8", page_icon="🌸", layout="centered")

# Custom CSS for the app's appearance
st.markdown("""
    <style>
        .title {
            color: #ff6f61;
            font-size: 36px;
            font-family: 'Roboto', sans-serif;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            color: #444;
            font-size: 18px;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin-bottom: 40px;
        }
        .stButton button {
            background-color: #ff6f61;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stFileUploader label {
            background-color: #ff6f61;
            color: white;
            border-radius: 5px;
            padding: 12px 20px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="title">🌸 Flower Detection with YOLOv8 🌸</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image of a flower, and YOLOv8 will predict the flower type and display the detected objects with confidence scores.</div>', unsafe_allow_html=True)

# File uploader with a colorful button
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the image with a border and drop shadow effect
    st.image(image, caption="Uploaded Image", use_column_width=True, clamp=True)

    # Convert the image to OpenCV format (for YOLOv8)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference on the image
    st.write("Running inference on the image... Please wait.")
    results = model(img_bgr)

    # Check if the results are in a list format and extract predictions
    if isinstance(results, list):
        result = results[0]  # Take the first item in the list (the actual results)
        boxes = result.boxes  # The predictions

        # Display predictions with a nice heading
        st.markdown("<h3 style='color:#ff6f61;'>Predictions:</h3>", unsafe_allow_html=True)
        for box in boxes:
            class_id = int(box.cls)  # Class ID
            confidence = box.conf.item()  # Convert tensor to scalar value
            class_name = model.names[class_id]  # Class name

            st.write(f"**{class_name}** - Confidence: {confidence:.2f}")

        # Display the image with bounding boxes
        annotated_image = result.plot()  # The image with detections
        st.image(annotated_image, caption="Predicted Image with Bounding Boxes", use_column_width=True, clamp=True)

        # Optionally, you can save the predictions
        output_image_path = "/tmp/predicted_image.jpg"
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, annotated_image)

        # Provide a download button for the user to download the predicted image
        st.download_button(
            "Download Predicted Image",
            data=open(output_image_path, "rb"),
            file_name="predicted_image.jpg",
            help="Click to download the predicted image with bounding boxes."
        )
