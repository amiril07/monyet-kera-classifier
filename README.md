# 🐒🦍 Monkey vs Ape Classifier Web App

A full-stack Deep Learning web application that classifies primate images into either **Monkey (Monyet)** or **Ape (Kera)**. This project utilizes **PyTorch** for the computer vision backbone and **Flask** for the minimalist web interface.

---

## 📌 Introduction
Distinguishing between monkeys and apes can be visually challenging due to overlapping features like fur patterns and habitats. This repository provides an end-to-end implementation of an image classifier using **Transfer Learning** with a pre-trained ResNet18 network to achieve robust classification with a small custom dataset.

---

## 🧠 Model Architecture & Mechanism

Unlike models trained completely from scratch, this project implements **Transfer Learning**. We leverage **ResNet18**, a deep convolutional neural network pre-trained on the massive ImageNet dataset, and fine-tune its final layers for our specific task.

The image processing flow follows a structured Deep Learning pipeline:

---

### 1. Feature Extraction (Convolutional Layers)
* **Convolution & Pooling:** The pre-trained layers filter low-level features (edges, textures) and high-level abstract shapes (facial structures, limb proportions) from the primate images.
* **Normalization:** Input images are resized to $224 \times 224$ pixels and normalized matching the ImageNet channels:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 2. Custom Classification (Fully-Connected Layer)
* **Flatten Layer:** The 2D spatial feature maps are flattened into a 1D vector.
* **Linear Classifier:** The original 1000-class output layer of ResNet18 is replaced with a custom Linear Layer mapping to **2 classes**:
  $$\text{Output} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}$$
  Where $\text{Output} \in \mathbb{R}^2$ representing raw scores for `[Kera, Monyet]`.
* **Softmax Activation:** To output a human-readable confidence score, the server applies a Softmax function on the raw outputs:
  $$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

---

## 🛠️ Tech Stack & Dependencies
* **Core AI Framework:** PyTorch (`torch`, `torchvision`)
* **Web Framework:** Flask (Python backend)
* **Frontend:** HTML5, JavaScript, Tailwind CSS
* **Image Processing:** Pillow (PIL)

---
