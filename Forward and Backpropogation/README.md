# CIFAR-10 Image Classification using Neural Networks

A deep learning project that implements a **fully connected neural network** using **TensorFlow/Keras** to classify images from the **CIFAR-10 dataset**.

The project focuses on understanding how important training parameters such as the **learning rate** and **number of epochs** affect model performance. It also goes beyond basic accuracy by analyzing generalization, per-class performance, prediction confidence, misclassified images, and the confusion matrix.

---

## Project Overview

The **CIFAR-10 dataset** contains **60,000 color images** belonging to **10 different classes**.

Each image has dimensions:

```text
32 × 32 × 3
```

where the three channels represent RGB color information.

Since the project uses fully connected Dense layers rather than convolutional layers, each image is flattened into a vector of:

```text
32 × 32 × 3 = 3072 features
```

The neural network then learns to classify each image into one of the ten CIFAR-10 categories.

---

## CIFAR-10 Classes

The ten classes used in the project are:

| Label | Class |
|------:|-------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

---

## Machine Learning Pipeline

```text
CIFAR-10 Dataset
       ↓
Load Training & Test Images
       ↓
Flatten 32×32×3 Images
       ↓
Normalize Pixel Values
       ↓
Create Fully Connected Neural Network
       ↓
Experiment with Learning Rates
       ↓
Experiment with Training Epochs
       ↓
Train Final Model
       ↓
Evaluate on Test Dataset
       ↓
Analyze Predictions
       ↓
Generalization Gap
       ↓
Per-Class Accuracy
       ↓
Prediction Confidence
       ↓
Confusion Matrix
       ↓
Analyze Confidently Misclassified Images
```

---

## Dataset

The CIFAR-10 dataset is loaded directly using TensorFlow/Keras:

```python
tf.keras.datasets.cifar10.load_data()
```

It contains:

```text
60,000 color images
10 classes
32 × 32 pixel resolution
```

The dataset is already divided into training and testing sets.

The class labels are flattened into one-dimensional arrays before training.

---

## Image Preprocessing

### Flattening

The original images have the shape:

```text
32 × 32 × 3
```

Since the model uses fully connected Dense layers, each image is reshaped into a single vector:

```text
32 × 32 × 3 = 3072
```

This results in an input shape of:

```text
3072 features
```

The transformation is performed using:

```python
x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)
```

---

## Pixel Normalization

The original image pixel values range from:

```text
0 – 255
```

They are divided by `255.0`:

```python
x_train = x_train.reshape(-1, 3072) / 255.0
x_test = x_test.reshape(-1, 3072) / 255.0
```

This converts the pixel values into the range:

```text
0 – 1
```

Normalization helps provide more suitable input values for neural network training.

---

## Neural Network Architecture

The project uses a fully connected neural network created with Keras' `Sequential` API.

```text
Input
3072 Features
     ↓
Dense Layer
256 Neurons
ReLU
     ↓
Dense Layer
128 Neurons
ReLU
     ↓
Output Layer
10 Neurons
Softmax
```

### Architecture Summary

| Layer | Configuration |
|-------|---------------|
| Input | 3072 features |
| Hidden Layer 1 | 256 neurons, ReLU |
| Hidden Layer 2 | 128 neurons, ReLU |
| Output Layer | 10 neurons, Softmax |

---

## Hidden Layer 1

The first Dense layer contains:

```python
Dense(256, activation='relu')
```

It contains **256 neurons** and uses the ReLU activation function.

The layer receives the 3072 pixel features and learns combinations of those features that can help distinguish between image classes.

---

## Hidden Layer 2

The second Dense layer contains:

```python
Dense(128, activation='relu')
```

It contains **128 neurons**.

This layer further combines the learned representations from the previous layer before passing them to the final classifier.

---

## Output Layer

The output layer contains:

```python
Dense(10, activation='softmax')
```

There are 10 neurons because CIFAR-10 contains 10 classes.

The Softmax activation converts the outputs into probabilities across the ten classes.

The class with the highest probability becomes the predicted class.

---

## Activation Function

### ReLU

The hidden layers use:

```text
ReLU
```

ReLU introduces non-linearity into the network, allowing it to learn more complex relationships between input features.

### Softmax

The output layer uses:

```text
Softmax
```

Softmax produces a probability distribution across the ten CIFAR-10 classes.

For example:

```text
Airplane    → 0.05
Automobile  → 0.10
Bird        → 0.03
Cat         → 0.20
...
Dog         → 0.50
...
```

The class with the highest probability is selected as the prediction.

---

## Model Compilation

The model is compiled using:

```python
m.compile(
    optimizer=Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

The project uses the **Adam optimizer**.

The learning rate is passed into the model so that different learning rates can be tested.

### Loss Function

The model uses:

```text
Sparse Categorical Cross-Entropy
```

This is suitable for multi-class classification where class labels are represented as integers.

### Metric

The main evaluation metric is:

```text
Accuracy
```

which measures the proportion of correctly classified images.

---

# Learning Rate Experiment

One of the main goals of the project is to understand how the **learning rate affects model performance**.

The following learning rates are tested:

```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
```

A separate model is created and trained for each learning rate.

Each model is trained for:

```text
5 epochs
Batch Size = 128
```

The resulting test accuracy is stored for comparison.

---

## Why Experiment with Learning Rate?

The learning rate controls how much the model's weights are changed during optimization.

A learning rate that is too high can cause unstable training or prevent the model from reaching a good solution.

A learning rate that is too low can make training unnecessarily slow.

Testing multiple values helps identify a learning rate that provides better performance for this particular model and dataset.

The experiment results are stored in a Pandas DataFrame and visualized using a logarithmic x-axis.

---

# Epoch Experiment

The project also investigates how the number of training epochs affects performance.

The following epoch configurations are tested:

```python
epochs_list = [5, 10, 20]
```

For each configuration, a new model is trained using:

```text
Learning Rate = 0.001
Batch Size = 128
```

The test accuracy for each configuration is then recorded.

---

## Why Experiment with Epochs?

An epoch represents one complete pass through the training dataset.

Training for too few epochs may cause the model to underfit because it has not had enough opportunities to learn.

Training for too many epochs can lead to overfitting, where the model performs well on training data but becomes worse at generalizing to unseen data.

The experiment helps observe how performance changes as training time increases.

---

# Final Model

After the parameter experiments, the final model is created using:

```text
Learning Rate = 0.001
Epochs = 20
Batch Size = 128
```

The model is trained with a validation split:

```python
validation_split=0.2
```

This means 20% of the training data is used to monitor validation performance during training.

---

## Final Training

The final model is trained using:

```python
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128
)
```

The training history stores information about:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These values are later used to analyze model performance.

---

# Training and Validation Accuracy

The project plots:

```python
history.history['accuracy']
history.history['val_accuracy']
```

The two curves allow comparison between:

```text
Training Accuracy
        vs.
Validation Accuracy
```

If training accuracy continues increasing while validation accuracy stops improving or decreases, this may indicate overfitting.

---

# Training and Validation Loss

The project also plots:

```python
history.history['loss']
history.history['val_loss']
```

These curves show how the model's prediction error changes throughout training.

A decreasing training loss generally indicates that the model is learning from the training data.

Comparing training and validation loss can also help identify potential overfitting.

---

# Test Evaluation

After training, the final model is evaluated using the unseen CIFAR-10 test dataset:

```python
model.evaluate(x_test, y_test)
```

This provides the final:

```text
Test Loss
Test Accuracy
```

The test dataset was not used to train the final model, making it useful for measuring how well the model generalizes to unseen images.

---

# Prediction Analysis

The trained model generates predictions for all test images:

```python
predictions = model.predict(x_test)
```

The model returns probabilities for all ten classes.

The predicted class is selected using:

```python
y_pred = np.argmax(predictions, axis=1)
```

This selects the class with the highest predicted probability.

---

# Incorrect Predictions

The project identifies incorrectly classified images using:

```python
wrong = np.where(y_pred != y_test)[0]
```

These images are displayed along with:

```text
True Class
Predicted Class
```

This makes it possible to visually inspect examples where the model struggles.

---

# Generalization Gap

The project calculates the **generalization gap** using the difference between training and validation accuracy:

```python
gap = np.array(train_acc) - np.array(val_acc)
```

The gap is plotted across the 20 training epochs.

### Why is this useful?

A large difference between training and validation accuracy can indicate that the model is learning the training data much better than unseen validation data.

This can be a sign of:

```text
Overfitting
```

The generalization-gap plot therefore provides another way to analyze how well the model generalizes during training.

---

# Per-Class Accuracy

Overall accuracy does not show whether every class is being classified equally well.

The project calculates accuracy separately for each of the ten CIFAR-10 classes.

For every class, it:

1. Finds the test images belonging to that class.
2. Compares the predicted labels with the actual labels.
3. Calculates the average classification accuracy.

The results are visualized using a bar chart.

---

## Why Per-Class Accuracy Matters

A model can have reasonable overall accuracy while still performing poorly on particular classes.

For example, visually similar classes such as:

```text
Cat vs Dog
Automobile vs Truck
```

may be more difficult for the model to distinguish.

Per-class analysis helps identify these differences.

---

# Prediction Confidence

The model's prediction confidence is calculated using:

```python
confidence = np.max(predictions, axis=1)
```

This takes the highest probability produced by the Softmax output for each image.

For example:

```text
Prediction Probability:
0.05
0.10
0.02
0.73
0.10
...
```

The model's confidence for that prediction would be:

```text
73%
```

---

# Confidence Distribution

The project compares confidence values for:

```text
Correct Predictions
vs.
Incorrect Predictions
```

using histograms.

This helps answer an important question:

> Is the model uncertain when it makes mistakes, or does it sometimes make incorrect predictions with high confidence?

Understanding confidence can provide additional insight into model reliability.

---

# Confusion Matrix

A confusion matrix is generated using:

```python
confusion_matrix(
    y_test,
    y_pred
)
```

The matrix compares:

```text
Actual Class
      vs.
Predicted Class
```

It provides a class-by-class view of the model's performance.

---

## What the Confusion Matrix Shows

The diagonal values represent correctly classified images.

Off-diagonal values represent incorrect classifications.

For example:

```text
Actual Cat → Predicted Dog
```

would appear as an error between the Cat and Dog classes.

This makes it easier to identify which classes the model commonly confuses.

---

# Confidently Misclassified Images

The project goes one step further by finding incorrect predictions where the model had high confidence.

The incorrectly classified images are ranked according to their prediction confidence.

The top 12 confidently misclassified images are displayed with:

```text
True Class
Predicted Class
Confidence
```

These examples are particularly useful because they show cases where the model was strongly convinced about an incorrect prediction.

---

# Why Analyze Confident Mistakes?

A model being uncertain about a difficult image is different from being highly confident and wrong.

Confident mistakes can reveal:

- Visually similar classes
- Ambiguous images
- Dataset limitations
- Features the model may be relying on incorrectly
- Areas where the model architecture could be improved

This type of analysis provides more insight than accuracy alone.

---

# Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Deep Learning**
- **Neural Networks**
- **Image Classification**

---

# Key Concepts Demonstrated

This project demonstrates practical understanding of:

- CIFAR-10 Dataset
- Image Classification
- Neural Networks
- Fully Connected Networks
- Dense Layers
- Image Flattening
- Pixel Normalization
- ReLU Activation
- Softmax Activation
- Forward Propagation
- Backpropagation
- Adam Optimization
- Learning Rate
- Epochs
- Batch Size
- Training Data
- Validation Data
- Test Data
- Model Generalization
- Overfitting
- Generalization Gap
- Prediction Confidence
- Per-Class Accuracy
- Confusion Matrix
- Misclassification Analysis

---

# Project Structure

```text
CIFAR10-Neural-Network/
│
├── CIFAR10_Neural_Network.ipynb
│
└── README.md
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CIFAR10-Neural-Network
```

## 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the CIFAR-10 notebook and run the cells sequentially.

The CIFAR-10 dataset will be downloaded automatically through TensorFlow/Keras.

---

# Results

The notebook evaluates the neural network using several different perspectives.

### Hyperparameter Experiments

The project compares:

```text
Learning Rates:
0.1
0.01
0.001
0.0001
```

and:

```text
Epochs:
5
10
20
```

### Model Performance Analysis

The final model is analyzed using:

- Test accuracy
- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Generalization gap
- Per-class accuracy
- Prediction confidence
- Confusion matrix
- Incorrect predictions
- Highly confident incorrect predictions

The actual numerical results depend on the training run and should be taken directly from the notebook output.

---

# Limitations

The main limitation of this project is that it uses a **fully connected neural network** rather than a Convolutional Neural Network.

Flattening a CIFAR-10 image from:

```text
32 × 32 × 3
```

into:

```text
3072 features
```

removes the explicit spatial structure of the image.

This makes the approach less suitable for complex image recognition compared with CNN-based architectures.

The model may therefore struggle with visually similar classes such as:

```text
Cat / Dog
Automobile / Truck
Bird / Airplane
```

The dataset and model configuration also limit the achievable performance.

---

# Future Improvements

The project can be improved by:

- Replacing the Dense network with a CNN
- Using convolutional layers for spatial feature extraction
- Adding Batch Normalization
- Adding Dropout
- Using data augmentation
- Performing more extensive hyperparameter tuning
- Testing additional learning rates
- Testing different batch sizes
- Using learning-rate scheduling
- Implementing Early Stopping
- Using Transfer Learning
- Comparing architectures such as ResNet or EfficientNet
- Performing more detailed error analysis
- Calibrating prediction confidence
- Deploying the trained model as an application

---

# Key Takeaway

This project demonstrates how a **fully connected neural network can be applied to image classification using the CIFAR-10 dataset**, while also showing why model evaluation should go beyond a single accuracy value.

The project explores:

```text
Hyperparameter Selection
        ↓
Model Training
        ↓
Validation
        ↓
Test Evaluation
        ↓
Generalization Analysis
        ↓
Class-Level Analysis
        ↓
Confidence Analysis
        ↓
Error Analysis
```

The analysis of **learning rate, epochs, generalization gap, per-class accuracy, prediction confidence, confusion matrix, and confidently misclassified images** provides a more complete understanding of how the neural network behaves.

It also establishes a useful foundation for moving from basic fully connected networks toward **CNN-based computer vision models**.

---

# Author

**Lavanya**

Machine Learning | Deep Learning | Computer Vision | Neural Networks
