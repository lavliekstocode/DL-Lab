# MNIST Handwritten Digit Classification using Neural Networks

A deep learning project that uses a **fully connected neural network built with TensorFlow/Keras** to recognize handwritten digits from the **MNIST dataset**.

The project demonstrates an end-to-end image classification workflow, including dataset loading, image normalization, neural network construction, model training, evaluation, visualization, and model saving.

---

## Project Overview

The **MNIST dataset** contains grayscale images of handwritten digits ranging from **0 to 9**.

Each image has dimensions:

```text
28 × 28 pixels
```

The objective of this project is to train a neural network that can correctly classify each handwritten image into one of the ten digit classes:

```text
0  1  2  3  4  5  6  7  8  9
```

The model learns patterns from the training images and uses those learned patterns to predict previously unseen handwritten digits.

---

## Machine Learning Pipeline

```text
MNIST Dataset
      ↓
Load Training & Test Data
      ↓
Pixel Normalization
      ↓
Flatten 28×28 Images
      ↓
Dense Neural Network
      ↓
ReLU Activation
      ↓
Softmax Classification
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Training Performance Visualization
      ↓
Save Trained Model
```

---

## Dataset

The project uses the **MNIST handwritten digit dataset** provided through TensorFlow/Keras.

The images are:

- Grayscale
- 28 × 28 pixels
- Classified into 10 digit categories
- Represent handwritten digits from 0 to 9

The dataset is automatically loaded using:

```python
tf.keras.datasets.mnist.load_data()
```

The dataset is divided into training and testing sets.

---

## Data Preprocessing

### Pixel Normalization

The original pixel values range from:

```text
0 – 255
```

They are normalized to:

```text
0 – 1
```

using:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Normalization provides smaller and more consistent input values for the neural network.

---

## Neural Network Architecture

The project uses a simple **fully connected neural network** implemented using Keras' Sequential API.

```text
Input Image
28 × 28
   ↓
Flatten
784 Features
   ↓
Dense Layer
128 Neurons
ReLU
   ↓
Dense Output Layer
10 Neurons
Softmax
   ↓
Predicted Digit
```

### Model Configuration

| Component | Configuration |
|------------|------------|
| Input Size | 28 × 28 |
| Flattened Input | 784 features |
| Hidden Layer | 128 neurons |
| Hidden Activation | ReLU |
| Output Layer | 10 neurons |
| Output Activation | Softmax |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Training Epochs | 3 |

---

## Why Flatten the Images?

The original MNIST images are two-dimensional:

```text
28 × 28
```

The neural network uses Dense layers, so the images are flattened into a single vector containing:

```text
28 × 28 = 784 values
```

The `Flatten` layer performs this conversion automatically before the image is passed to the Dense layer.

---

## Activation Functions

### ReLU

The hidden layer uses the **Rectified Linear Unit (ReLU)** activation function.

ReLU introduces non-linearity into the network, allowing it to learn more complex patterns from the handwritten digits.

### Softmax

The output layer contains **10 neurons**, representing the ten possible digits.

Softmax converts the output values into probabilities for each class.

The digit with the highest probability becomes the model's prediction.

---

## Model Compilation

The model is compiled using:

```python
my_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

**Adam** is used to update the model's weights during training.

### Loss Function

**Sparse Categorical Cross-Entropy** is used because the task is a multi-class classification problem where each image belongs to one digit class.

### Evaluation Metric

**Accuracy** measures the percentage of correctly classified images.

---

## Model Training

The neural network is trained using:

```text
Epochs = 3
```

During training, the model processes the training images and repeatedly updates its weights to reduce classification error.

The training process follows:

```text
Input Image
    ↓
Forward Pass
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Weight Update
    ↓
Next Training Step
```

This process allows the neural network to gradually learn visual patterns associated with different handwritten digits.

---

## Model Evaluation

After training, the model is evaluated using the MNIST test dataset.

The test dataset contains images that the model did not use during training.

The evaluation returns:

- Test Loss
- Test Accuracy

The test accuracy provides an estimate of how well the trained model generalizes to unseen handwritten digits.

---

## Data Visualization

The project visualizes individual MNIST images using Matplotlib.

A sample image can be displayed using:

```python
plt.imshow(train_images[1], cmap='gray')
```

The notebook also displays a grid containing multiple handwritten digit examples.

This provides a visual understanding of the input data before and during model development.

---

## Training Accuracy Visualization

The project includes a training accuracy plot showing how the model's accuracy changes across epochs.

```text
Epoch
  ↓
Training Accuracy
```

This helps visualize whether the model is improving as training progresses.

---

## Training Loss Visualization

The project also plots the training loss across epochs.

```text
Epoch
  ↓
Training Loss
```

A decreasing training loss generally indicates that the model is learning to reduce its prediction error on the training data.

---

## Model Saving

After training, the model is saved in Keras format:

```python
my_model.save('my_mnist_model.keras')
```

The resulting file can be loaded later without retraining the neural network.

This makes it possible to reuse the trained model for future digit predictions or integrate it into another application.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Matplotlib
- Deep Learning
- Artificial Neural Networks
- Image Classification

---

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Neural Networks
- Fully Connected Layers
- Image Classification
- MNIST Dataset
- Data Normalization
- Image Flattening
- ReLU Activation
- Softmax Activation
- Forward Propagation
- Backpropagation
- Adam Optimization
- Cross-Entropy Loss
- Model Training
- Model Evaluation
- Training Accuracy
- Training Loss
- Model Persistence

---

## Project Structure

```text
MNIST-Neural-Network/
│
├── MNIST_Neural_Network.ipynb
│
├── my_mnist_model.keras
│
└── README.md
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MNIST-Neural-Network
```

### 2. Install Dependencies

```bash
pip install tensorflow matplotlib
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
MNIST_Neural_Network.ipynb
```

Run the cells sequentially.

The MNIST dataset will be downloaded automatically through TensorFlow/Keras.

---

## Results

The notebook evaluates the trained model on the MNIST test dataset and reports its test accuracy.

It also generates visualizations for:

- Sample handwritten digits
- Multiple training images
- Training accuracy across epochs
- Training loss across epochs

These visualizations provide both quantitative and visual insight into the model's learning process.

---

## Limitations

The current project uses a **fully connected neural network**.

Although this architecture works well for MNIST, flattening the images removes their original two-dimensional spatial structure.

For more complex image classification tasks, a **Convolutional Neural Network (CNN)** would generally be more suitable because CNNs can learn spatial and local visual features directly from images.

---

## Future Improvements

The project can be extended in several ways:

- Increase the number of training epochs
- Add additional hidden layers
- Experiment with different numbers of neurons
- Add Dropout for regularization
- Add Batch Normalization
- Implement a CNN-based architecture
- Compare different optimizers
- Perform hyperparameter tuning
- Add a confusion matrix
- Analyze incorrect predictions
- Build an interface for uploading handwritten digits
- Deploy the trained model as a web application

---

## Key Takeaway

This project provides a practical introduction to building a neural network for image classification using **TensorFlow and Keras**.

The complete workflow covers:

**Load → Preprocess → Build → Train → Evaluate → Visualize → Save**

It establishes the fundamental concepts required for moving from basic neural networks toward more advanced deep learning and computer vision architectures.

---

## Author

**Lavanya**

Machine Learning | Deep Learning | Neural Networks | Computer Vision
