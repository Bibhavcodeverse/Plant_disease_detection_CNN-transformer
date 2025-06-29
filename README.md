# Plant_disease_detection_CNN-transformer

# 🌿 CNN + Transformer Hybrid Model for Plant Disease Detection

This project implements a hybrid deep learning model combining **Convolutional Neural Networks (CNNs)** and **Transformers** to detect plant diseases from leaf images.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.ipynb` | Jupyter notebook containing data preprocessing, model architecture, training, and evaluation |
| `cnn_transformer_full_model.pth` | Saved PyTorch model weights (CNN + Transformer hybrid) |

---

## 🚀 Features

- 🌱 **Plant disease detection** using leaf images
- 🧠 **Hybrid model**: Combines CNN (for feature extraction) and Transformer (for global context)
- 💾 Includes **saved model file** for inference or fine-tuning
- 📊 Model performance evaluation (accuracy, loss curves, etc.)

---

## 🧠 Model Overview

### Architecture:
- **CNN Backbone**: Extracts local spatial features
- **Transformer Encoder**: Captures global dependencies and attention
- **Fully Connected Layers**: For classification of plant disease categories

---

## 📝 How to Use

1. Clone the repo and install dependencies:

pip install torch torchvision matplotlib
import torch
from model import YourModelClass  # Replace with your model class

model = YourModelClass()
model.load_state_dict(torch.load("cnn_transformer_full_model.pth"))
model.eval()


Dataset
You can use any labeled plant disease dataset, such as:

PlantVillage Dataset

📊 Sample Output
Accuracy: ~99.5% (Replace with actual)
![WhatsApp Image 2025-04-04 at 02 08 00_7bd5f58e](https://github.com/user-attachments/assets/588522fd-e190-4fe1-a15c-09c6cfa21d00)


Loss Graphs

Confusion Matrix

Visual predictions on test images

🛠️ Future Work
Add real-time webcam-based detection

Improve Transformer efficiency with Swin or MobileViT

Convert to ONNX or TensorRT for deployment
