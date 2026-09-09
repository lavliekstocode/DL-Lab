# CIFAR-10 Image Classification using a Neural Network

A deep learning project that implements a fully connected neural network for multi-class image classification on the **CIFAR-10 dataset**.

The project goes beyond basic model training by investigating how **learning rate, training duration, generalization, class-level performance, prediction confidence, and model errors** affect classification performance.

---

## Project Overview

The CIFAR-10 dataset contains **60,000 RGB images** of size 32×32 belonging to 10 different classes.

The ten classes are:

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

The images are flattened into 3072-dimensional feature vectors and passed through a fully connected neural network.

The project focuses not only on training a neural network, but also on **understanding its behavior, evaluating its generalization, and analyzing where and why it makes mistakes**.

---

## Machine Learning Pipeline

```text
CIFAR-10 Dataset
       ↓
Image Reshaping
       ↓
Pixel Normalization
       ↓
Fully Connected Neural Network
       ↓
Learning Rate Experiments
       ↓
Epoch Experiments
       ↓
Final Model Training
       ↓
Model Evaluation
       ↓
Error & Confidece Analysis

