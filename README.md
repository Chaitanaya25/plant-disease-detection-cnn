# 🌱 Plant Disease Detection using Convolutional Neural Networks (CNN)

A deep learning–based image classification project that detects plant diseases from leaf images using a **Convolutional Neural Network (CNN)**.  
The model predicts the disease class along with a **confidence percentage**.

---

## 🧠 Project Overview

Plant diseases can significantly impact crop yield and quality.  
This project uses a CNN trained on leaf images to automatically identify plant diseases and assist in early detection.

---

## 🚀 Key Features

- CNN-based plant disease classification
- Single-image inference pipeline
- Predicts disease name with confidence percentage
- Model evaluation using precision, recall, F1-score, and confusion matrix
- Clean and modular codebase

---

## 🖼️ Demo Results

### 🔹 Input Image
Leaf image given as input to the model:

![Input Image](Images/input.jpg)

---

### 🔹 Prediction Output
Predicted disease name with confidence percentage:

![Prediction Output](Images/output.jpg)

---

## 📊 Model Performance Visualizations

### 🔹 Precision vs Recall
Shows the trade-off between precision and recall across classes:

![Precision vs Recall](Images/pvsr.jpg)

---

### 🔹 F1 Score
F1 score visualization across disease classes:

![F1 Score](Images/f1.jpg)

---

### 🔹 Confusion Matrix
Heatmap representing classification performance across all classes:

![Confusion Matrix](Images/cm.jpg)

---

## 🧪 Model Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Image Size: 128 × 128 × 3
- Normalization: Pixel values scaled to [0,1]
- Output: Probability distribution over disease classes
- Prediction Method: Argmax of probabilities
- Confidence Score: Maximum probability × 100

---

