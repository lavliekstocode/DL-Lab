# Multilayer Perceptron (MLP) for Iris Classification

A deep learning classification project that implements a **Multilayer Perceptron (MLP) neural network** to classify Iris flowers into their respective species using measurable flower characteristics.

The project demonstrates a complete machine learning workflow — from data preprocessing and feature scaling to neural network training, evaluation, and visualization.

## Project Overview

The model uses the classic **Iris dataset**, which contains measurements of:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The objective is to classify each flower into one of three species:

* Setosa
* Versicolor
* Virginica

A neural network with **two hidden layers containing 10 neurons each** is trained using the Adam optimizer and ReLU activation.

## DL Pipeline

```text
Iris Dataset
     ↓
Train-Test Split
     ↓
Feature Standardization
     ↓
MLP Neural Network
     ↓
Model Training
     ↓
Class Prediction
     ↓
Performance Evaluation
     ↓
Visualization
```

## Model Architecture

The project uses Scikit-learn's `MLPClassifier` with the following configuration:

| Parameter          | Configuration         |
| ------------------ | --------------------- |
| Model              | Multilayer Perceptron |
| Hidden Layers      | 2                     |
| Neurons per Layer  | 10, 10                |
| Activation         | ReLU                  |
| Optimizer          | Adam                  |
| Maximum Iterations | 1000                  |
| Random State       | 42                    |
| Input Features     | 4                     |
| Output Classes     | 3                     |

## Data Preprocessing

Before training, the dataset is divided into:

* **80% training data**
* **20% testing data**

Feature standardization is performed using `StandardScaler`.

The scaler is fitted only on the training data and then applied to the test data:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

This prevents information from the test set from influencing the preprocessing stage.

## Model Evaluation

The trained neural network is evaluated using multiple metrics rather than relying only on accuracy.

### Accuracy

Measures the overall proportion of correctly classified samples.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each Iris species.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

This provides a more detailed view of model performance across individual classes.

## Visualizations

The notebook includes several visual analyses:

### Confusion Matrix

Provides a class-by-class view of prediction performance.

### Actual Iris Classes

A scatter plot is used to visualize the actual classes in the test dataset.

### Predicted Iris Classes

The model's predictions are visualized using the same feature space, allowing the actual and predicted distributions to be compared.

### Petal Feature Distribution

Petal length and petal width are plotted to visualize the separation between Iris species.

## Technologies Used

* **Python**
* **Scikit-learn**
* **Matplotlib**
* Machine Learning
* Artificial Neural Networks
* Multilayer Perceptron (MLP)

## Project Structure

```text
MLP-Iris-Classification/
│
├── Multilayer Perceptron (MLP) for classification of the Iris Dataset (1).ipynb
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd MLP-Iris-Classification
```

### 2. Install dependencies

```bash
pip install scikit-learn matplotlib
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `.ipynb` file and run the cells sequentially.

## Key Learning Outcomes

This project demonstrates practical understanding of:

* Supervised learning
* Multi-class classification
* Neural network architecture
* Multilayer Perceptrons
* Feature standardization
* Train-test splitting
* ReLU activation
* Adam optimization
* Model evaluation
* Confusion matrix analysis
* Classification metrics
* Data visualization

## Why This Project Matters

Although the Iris dataset is small and well-known, the project demonstrates the same fundamental workflow used in larger machine learning systems:

**preprocess → train → predict → evaluate → visualize**

The implementation provides a foundation for extending the approach to more complex classification problems and larger datasets.

## Future Improvements

Potential extensions include:

* Hyperparameter tuning for hidden-layer sizes and learning rate
* Cross-validation for more robust performance estimation
* Comparison with Logistic Regression, SVM, Random Forest, and other classifiers
* Experimenting with different activation functions
* Building a prediction interface for new flower measurements
* Tracking training performance across different neural-network configurations

## Author

**Lavanya**

This repository is part of a collection of machine learning implementations focused on developing practical understanding of neural networks, classification, and model evaluation.
