# COVID-19 Chest X-ray Classification using Deep Learning
**Objective**
This project aims to build an AI-powered classifier that can detect COVID-19, Viral Pneumonia, or Normal chest conditions from X-ray images using deep learning techniques.

Problem Statement
Early and accurate detection of COVID-19 through chest X-rays can assist healthcare professionals in faster diagnosis, especially in resource-limited settings. Radiological features of COVID-19 often overlap with other conditions like pneumonia, making automated classification highly valuable.

Tech Stack
Google Colab for training
Python, TensorFlow/Keras for model development
OpenCV, NumPy for image preprocessing
Streamlit for app deployment in PyCharm

Dataset
Source: COVID-19 Radiography Database
Classes:
COVID-19 Positive
Normal
Viral Pneumonia

Model Details
Input Size: 128x128 grayscale images
Model Type: Custom CNN (3 convolutional layers + batch normalization)
Output: 3-class softmax (COVID, Normal, Viral Pneumonia)
Accuracy: ~ 82%
Optimizer: Adam

Loss Function: Categorical Crossentropy

Features
Upload chest X-ray and classify into one of the 3 categories
See model confidence score
Simple and fast UI for real-time predictions

Future Enhancements
Add Grad-CAM visualization for heatmap localization
Improve accuracy with EfficientNet or pretrained models
Deploy publicly via Streamlit Cloud
