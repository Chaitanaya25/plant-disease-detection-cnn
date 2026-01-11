# 🌱 Plant Disease Detection using Convolutional Neural Networks (CNN)

A **high-performance deep learning image classification system** that detects **plant diseases from leaf images** using a **Convolutional Neural Network (CNN)**.
The model predicts the **disease class along with confidence percentage**, achieving **excellent accuracy and strong generalization across 38 classes**.

---

## 🧠 Project Overview

Plant diseases are a major cause of crop loss worldwide.
Manual inspection is **time-consuming, error-prone, and requires expert knowledge**.

This project leverages **deep learning and computer vision** to:

* Automatically detect plant diseases from leaf images
* Provide **fast, accurate, and scalable diagnosis**
* Assist farmers and researchers in **early disease detection and decision-making**

---

## 🚀 Key Features

* 🌿 CNN-based **multi-class plant disease classification**
* 🖼️ **Single-image inference** with confidence score
* 📊 Comprehensive evaluation:

  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
* 📈 **Per-class performance analysis (38 classes)**
* 🧪 Strong generalization with minimal overfitting
* 🧱 Clean, modular, and reproducible training pipeline

---

## 📂 Dataset & Classes

* Leaf images representing **38 plant disease and healthy classes**
* Images resized and normalized for efficient training
* Balanced learning supported by strong per-class metrics

---

## 🧩 Model Architecture

* **Architecture**: Convolutional Neural Network (CNN)
* **Input Shape**: `128 × 128 × 3`
* **Normalization**: Pixel values scaled to `[0, 1]`
* **Output Layer**: Softmax over 38 classes
* **Prediction Logic**:

  * `Predicted Class = argmax(probabilities)`
  * `Confidence (%) = max(probability) × 100`

---

## 🏋️ Training Strategy

* Progressive training with **learning-rate scheduling**
* Validation-based **model checkpointing**
* Early stopping logic based on **best validation accuracy**
* Careful regularization to prevent overfitting

---

## 📈 Training Performance

### 🔹 Selected Epoch Highlights

| Epoch | Train Accuracy | Validation Accuracy | Status               |
| ----: | -------------: | ------------------: | -------------------- |
|    10 |         99.68% |              99.41% | ✅ Best (Initial)     |
|    11 |         99.74% |              99.40% | No improvement       |
|    12 |         99.74% |              99.35% | No improvement       |
|    13 |         99.77% |              99.45% | ✅ **New Best**       |
|    15 |         99.75% |              99.37% | No improvement       |


---

## 📊 Evaluation Metrics & Visualizations

### 🔹 Confusion Matrix

Shows excellent diagonal dominance, indicating **very low misclassification across classes**.

![Confusion Matrix](Images/cm.png)

---

### 🔹 Per-Class F1 Score

Nearly all classes achieve **F1-scores close to 1.0**, reflecting balanced precision and recall.

![F1 Score](Images/f1.png)

---

### 🔹 Precision vs Recall

Demonstrates **tight clustering near (1.0, 1.0)**, confirming strong model reliability.

![Precision vs Recall](Images/pvsr.png)

---

## 🧪 Final Results

```text
Training Accuracy   : 99.94%
Training Loss       : 0.0022
Validation Accuracy : 99.19%
Validation Loss     : 0.0251
```

✅ **High accuracy with minimal generalization gap**
✅ **Stable validation performance**
✅ **Production-ready model behavior**

---

## 🔍 Inference Pipeline

1. Load trained CNN model
2. Preprocess input image:

   * Resize to `128 × 128`
   * Normalize pixel values
3. Run forward pass
4. Extract:

   * Predicted disease class
   * Confidence percentage
5. Display result

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **Deep Learning Framework**: TensorFlow / Keras
* **Libraries**:

  * NumPy
  * Matplotlib
  * Scikit-learn
* **Visualization**: Confusion Matrix, F1 Score, Precision-Recall

---

## 🔮 Future Improvements

* 🌍 Deploy as a **web or mobile application**
* 📷 Real-time disease detection using camera input
* 🧠 Experiment with **transfer learning (ResNet, EfficientNet)**
* 🌾 Expand dataset for more crop varieties
* ☁️ Cloud deployment with REST API

---

## ⭐ Key Takeaways

* Achieved **>99% accuracy** on a challenging multi-class dataset
* Strong per-class performance across **38 disease categories**
* Robust evaluation confirms **real-world usability**
* Clean, scalable, and extensible deep learning pipeline

---

