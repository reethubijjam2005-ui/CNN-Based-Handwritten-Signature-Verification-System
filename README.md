 Overview

DeepSign is a deep learning-based system that detects whether a handwritten signature is genuine or forged using a Convolutional Neural Network (CNN).
The model is trained on a real-world dataset and can classify signature images with good accuracy.

This project demonstrates the application of Computer Vision + Deep Learning for authentication systems.

 Objective
Detect forged signatures automatically
Build a CNN model for image classification
Provide a simple pipeline from dataset → training → evaluation

 Tech Stack
Python
TensorFlow / Keras
OpenCV
NumPy
Matplotlib
Scikit-learn
KaggleHub (dataset)

 Dataset

Dataset used:
 Handwritten Signature Dataset from Kaggle

Genuine signatures
Forged signatures
Images in .png and .jpg format

 How It Works
1. Data Loading
Dataset downloaded using KaggleHub
Images resized to 128 × 128
Pixel values normalized
2. Labeling
0 → Genuine
1 → Forged (based on folder name)
3. Model Architecture
3 Convolutional Layers
MaxPooling Layers
Fully Connected Dense Layers
Dropout for regularization
Sigmoid output for binary classification

 Model Architecture (Summary)
Input Image (128x128x3)
↓
Conv2D (32) + ReLU
↓
MaxPooling
↓
Conv2D (64) + ReLU
↓
MaxPooling
↓
Conv2D (128) + ReLU
↓
MaxPooling
↓
Flatten
↓
Dense (128) + ReLU
↓
Dropout (0.5)
↓
Dense (1) + Sigmoid

 How to Run
Step 1 — Install Dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn kagglehub
Step 2 — Run the Script
python your_file_name.py

 Output
Training & validation accuracy graph
Final test accuracy printed
Saved model: signature_model.h5

Sample Result
Achieves good accuracy in distinguishing genuine vs forged signatures
Performance depends on dataset quality and size

 Features
✔ Automated signature verification
✔ Deep learning-based classification
✔ Easy to extend for real-world applications
✔ Lightweight CNN model

 Future Improvements
Use Transfer Learning (ResNet, VGG16)
Improve dataset balance
Add web interface (Flask / FastAPI)
Real-time signature upload & prediction
Data augmentation for better accuracy

 Applications
Banking authentication
Document verification
Fraud detection systems
Digital identity validation


