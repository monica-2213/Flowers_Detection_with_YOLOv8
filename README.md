# YOLOv8 Flower Detection

This repository contains a YOLOv8-based model trained to detect five types of flowers: Dandelion, Rose, Sunflower, Tulip, and Daisy. The model was trained on the [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset) and is designed to perform object detection by identifying and classifying the flower types.

## Streamlit App

You can interact with the trained model and predict the flower type from an image by using the [Streamlit app](https://flowersdetectionwithyolov8.streamlit.app/). Simply upload an image of a flower, and the model will predict the flower type with bounding boxes and confidence scores.

## Dataset

The dataset consists of 100 annotated images for each flower type:

- **Dandelion**
- **Rose**
- **Sunflower**
- **Tulip**
- **Daisy**

These images were manually annotated using **LabelImg** to create bounding boxes around the flowers for object detection.

## Model

The model was trained using **YOLOv8**, a state-of-the-art object detection algorithm, with the following training settings:

- **Batch size**: 16
- **Optimizer**: Adam
- **Epochs**: 100

### Results

After 100 epochs, the model achieved the following performance metrics:

- **mAP@50**: 0.813
- **mAP@50-95**: 0.61
- **Box Precision (P)**: 0.793
- **Recall (R)**: 0.812

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/monica-2213/Flowers_Detection_with_YOLOv8.git
   cd Flowers_Detection_with_YOLOv8
