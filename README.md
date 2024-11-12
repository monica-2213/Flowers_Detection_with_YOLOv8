# YOLOv8 Flower Detection

This repository contains a YOLOv8-based model trained to detect five types of flowers: Dandelion, Rose, Sunflower, Tulip, and Daisy. The images used for training were sourced from the [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset). 

## Dataset

The dataset consists of 100 annotated images for each flower type:

- Dandelion
- Rose
- Sunflower
- Tulip
- Daisy

These images were manually annotated using **LabelImg** to create bounding boxes around the flowers for object detection.

## Model

The model was trained using **YOLOv8**, which is a state-of-the-art object detection algorithm. The following settings were used for training:

- **Batch size**: 16
- **Optimizer**: Adam

### Results

After 100 epochs, the model achieved the following performance metrics:

- **mAP@50**: 0.813
- **mAP@50-95**: 0.61
- **Box Precision (P)**: 0.793
- **Recall (R)**: 0.812
