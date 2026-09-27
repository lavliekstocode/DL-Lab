# AlexNet vs VGG16 vs ResNet50 vs EfficientNetB0 on CIFAR-10

A deep learning project that compares four popular **Convolutional Neural Network (CNN) architectures** on the **CIFAR-10 image classification dataset**.

The project implements and evaluates:

- **AlexNet**
- **VGG16**
- **ResNet50**
- **EfficientNetB0**

The models are trained on a subset of CIFAR-10 and compared using **test accuracy, test loss, training time, validation accuracy, and validation loss**.

---

## Project Overview

The goal of this project is to understand how different CNN architectures perform on the same image classification task.

Instead of training only one model, the project creates four different architectures and evaluates them under the same general training setup.

```text
CIFAR-10 Dataset
       ↓
Select Training/Test Subsets
       ↓
Resize Images to 96 × 96
       ↓
Data Augmentation
       ↓
Create CNN Architectures
       ↓
AlexNet
VGG16
ResNet50
EfficientNetB0
       ↓
Compile Models
       ↓
Train for 5 Epochs
       ↓
Evaluate on Test Data
       ↓
Compare Accuracy, Loss & Training Time
       ↓
Analyze Validation Performance
       ↓
Identify Best Performing Model
```

---

## Dataset

The project uses the **CIFAR-10 dataset**, which contains color images belonging to 10 different classes.

The dataset is loaded using:

```python
tf.keras.datasets.cifar10.load_data()
```

The original dataset contains:

```text
60,000 images
10 classes
32 × 32 RGB images
```

The notebook uses a smaller subset to make the comparison between multiple deep learning architectures more manageable.

---

## Dataset Subset

The following number of images is selected:

```text
Training Images: 10,000
Testing Images: 2,000
```

This is controlled using:

```python
TRAIN_SIZE = 10000
TEST_SIZE = 2000
```

The subset is created by taking the first specified number of images from the original dataset.

---

## CIFAR-10 Classes

The dataset contains the following ten classes:

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

The labels are converted into one-hot encoded vectors using:

```python
to_categorical(y_train, 10)
```

This produces a format suitable for the categorical cross-entropy loss used by the models.

---

## Image Resizing

CIFAR-10 images originally have dimensions:

```text
32 × 32 × 3
```

The images are resized to:

```text
96 × 96 × 3
```

using:

```python
tf.image.resize(
    x_train,
    (IMG_SIZE, IMG_SIZE)
)
```

where:

```python
IMG_SIZE = 96
```

The larger input size is useful for the pretrained architectures used in this comparison.

---

## Data Type Conversion

After resizing, the images are converted to floating-point values:

```python
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
```

This ensures that the image tensors can be processed correctly by the neural networks.

---

## Data Augmentation

The project uses the following augmentation techniques:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

### Augmentation Techniques

| Technique | Purpose |
|-----------|---------|
| Random Flip | Creates horizontally flipped versions of images |
| Random Rotation | Slightly changes image orientation |
| Random Zoom | Creates variations at different scales |

Data augmentation gives the models more varied training examples and can help improve generalization.

---

## Data Pipeline

The training data is converted into a TensorFlow dataset:

```python
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)
```

The training pipeline then applies:

```text
Shuffle
   ↓
Batch
   ↓
Prefetch
```

with:

```python
BATCH_SIZE = 32
```

Prefetching allows TensorFlow to prepare upcoming batches while the model is processing the current batch.

The test dataset is batched and prefetched but is not shuffled.

---

# Model Architectures

The project compares four CNN architectures:

```text
AlexNet
   ↓
VGG16
   ↓
ResNet50
   ↓
EfficientNetB0
```

Each model has a different architectural design and represents a different stage in the development of modern deep learning.

---

# 1. AlexNet

AlexNet is implemented from scratch using Keras layers.

The architecture follows:

```text
Input
96 × 96 × 3
      ↓
Data Augmentation
      ↓
Conv2D — 64 Filters
      ↓
MaxPooling
      ↓
Conv2D — 192 Filters
      ↓
MaxPooling
      ↓
Conv2D — 384 Filters
      ↓
Conv2D — 256 Filters
      ↓
Conv2D — 256 Filters
      ↓
MaxPooling
      ↓
Global Average Pooling
      ↓
Dense — 256
      ↓
Dropout — 0.5
      ↓
Dense — 10
Softmax
```

---

## AlexNet Convolution Layers

The model begins with:

```python
layers.Conv2D(
    64,
    kernel_size=3,
    strides=1,
    padding="same",
    activation="relu"
)
```

This is followed by max pooling.

The next convolution layer contains:

```text
192 filters
```

The deeper layers progressively use:

```text
384 filters
256 filters
256 filters
```

This allows the network to learn increasingly complex visual representations.

---

## AlexNet Classification Section

Instead of using a large traditional fully connected section, this implementation uses:

```python
layers.GlobalAveragePooling2D()
```

followed by:

```python
layers.Dense(256, activation="relu")
```

and:

```python
layers.Dropout(0.5)
```

The final layer contains 10 neurons:

```python
layers.Dense(
    10,
    activation="softmax"
)
```

---

# 2. VGG16

VGG16 is loaded using the Keras Applications API:

```python
VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

The model uses **ImageNet pretrained weights**.

The original VGG16 classification head is removed using:

```python
include_top=False
```

---

## Transfer Learning with VGG16

The pretrained VGG16 base is frozen:

```python
base_model.trainable = False
```

This means its pretrained weights are not updated during training.

A new classification head is added:

```text
VGG16 Base
     ↓
Global Average Pooling
     ↓
Dense — 128
     ↓
Dropout — 0.5
     ↓
Dense — 10
Softmax
```

This allows the pretrained feature extractor to be used for the CIFAR-10 classification task.

---

# 3. ResNet50

ResNet50 is loaded using:

```python
ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

It also uses pretrained ImageNet weights.

The original classification head is removed.

The pretrained base is frozen using:

```python
base_model.trainable = False
```

---

## ResNet50 Classification Head

The ResNet50 backbone is followed by:

```text
Global Average Pooling
        ↓
Dense — 128
        ↓
Dropout — 0.5
        ↓
Dense — 10
        ↓
Softmax
```

The model therefore uses ResNet50 primarily as a pretrained feature extractor.

---

# 4. EfficientNetB0

EfficientNetB0 is loaded using:

```python
EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

The model uses ImageNet pretrained weights.

The pretrained base is frozen:

```python
base_model.trainable = False
```

---

## EfficientNetB0 Classification Head

The architecture follows:

```text
EfficientNetB0 Base
        ↓
Global Average Pooling
        ↓
Dense — 128
        ↓
Dropout — 0.5
        ↓
Dense — 10
        ↓
Softmax
```

---

# Transfer Learning

VGG16, ResNet50, and EfficientNetB0 use **transfer learning**.

Instead of training the entire networks from scratch, pretrained ImageNet models are used as feature extractors.

```text
ImageNet Pretrained Model
          ↓
Freeze Base Network
          ↓
Add New Classification Head
          ↓
Train on CIFAR-10
```

The pretrained base models are frozen using:

```python
base_model.trainable = False
```

This reduces the number of parameters that need to be updated during training.

---

# Model Compilation

All four models are compiled using the same optimizer configuration:

```python
optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001
)
```

The loss function is:

```python
categorical_crossentropy
```

and the evaluation metric is:

```text
Accuracy
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 5 |
| Image Size | 96 × 96 |
| Number of Classes | 10 |

Using the same training configuration makes the model comparison more consistent.

---

# Training the Models

A reusable function is created:

```python
train_model(model, model_name)
```

The function:

1. Starts a timer.
2. Trains the model.
3. Evaluates it on the test dataset.
4. Calculates training time.
5. Prints test accuracy.
6. Prints test loss.
7. Returns the training history and evaluation results.

---

## Training Time

Training time is measured using:

```python
start_time = time.time()
```

and:

```python
training_time = time.time() - start_time
```

This allows the project to compare not only model accuracy but also computational cost.

---

# Model Evaluation

After training, each model is evaluated using:

```python
model.evaluate(
    test_ds,
    verbose=0
)
```

The following metrics are recorded:

```text
Test Accuracy
Test Loss
Training Time
```

These results are stored in a Pandas DataFrame.

---

# Model Comparison

The project creates a comparison table containing:

| Model | Test Accuracy (%) | Test Loss | Training Time |
|-------|-------------------|-----------|---------------|
| AlexNet | Recorded from run | Recorded from run | Recorded from run |
| VGG16 | Recorded from run | Recorded from run | Recorded from run |
| ResNet50 | Recorded from run | Recorded from run | Recorded from run |
| EfficientNetB0 | Recorded from run | Recorded from run | Recorded from run |

The actual values depend on the execution environment and training run.

---

# Validation Accuracy Comparison

The validation accuracy histories of all four models are plotted together.

```text
Epoch
  ↓
Validation Accuracy
  ↓
AlexNet
VGG16
ResNet50
EfficientNetB0
```

This makes it possible to compare how quickly each architecture learns and how its validation performance changes across the five epochs.

---

# Validation Loss Comparison

The project also compares validation loss:

```text
Epoch
  ↓
Validation Loss
  ↓
AlexNet
VGG16
ResNet50
EfficientNetB0
```

This helps visualize how the models' prediction errors change throughout training.

---

# Test Accuracy Comparison

A bar chart is generated using:

```python
plt.bar(
    results["Model"],
    results["Test Accuracy (%)"]
)
```

This provides a simple visual comparison of the final test accuracy achieved by each architecture.

---

# Best Model

The best-performing model is identified automatically using:

```python
best_model = results.loc[
    results["Test Accuracy (%)"].idxmax()
]
```

The notebook then prints:

```text
BEST MODEL

Model: <model name>
Accuracy: <accuracy> %
```

This avoids manually selecting the best model based on the results.

---

# Why Compare Multiple Architectures?

Different CNN architectures make different design choices.

### AlexNet

Provides a relatively straightforward CNN architecture and serves as a useful baseline.

### VGG16

Uses a deeper architecture with repeated convolutional blocks and benefits from ImageNet pretraining.

### ResNet50

Uses residual connections in its original architecture, allowing much deeper networks to be trained effectively.

### EfficientNetB0

Uses an architecture designed to achieve a strong balance between model size and performance.

Comparing them provides practical experience with different approaches to image classification.

---

# Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **CIFAR-10**
- **AlexNet**
- **VGG16**
- **ResNet50**
- **EfficientNetB0**
- **Transfer Learning**
- **Deep Learning**
- **Computer Vision**

---

# Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Convolutional Neural Networks
- Image Classification
- CIFAR-10
- Image Resizing
- Data Augmentation
- TensorFlow Data Pipelines
- Batch Processing
- Prefetching
- CNN Feature Extraction
- Transfer Learning
- ImageNet Pretrained Models
- Frozen Layers
- Global Average Pooling
- Dense Layers
- Dropout
- ReLU Activation
- Softmax Classification
- Categorical Cross-Entropy
- Adam Optimization
- Model Evaluation
- Validation Accuracy
- Validation Loss
- Test Accuracy
- Test Loss
- Training Time
- Model Comparison
- Error/Performance Analysis

---

# Project Structure

```text
CNN-Architecture-Comparison/
│
├── Alexnet, ResNet, VGG, EfficientNet (1).ipynb
│
└── README.md
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CNN-Architecture-Comparison
```

## 2. Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

## 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

The CIFAR-10 dataset will be downloaded automatically through TensorFlow/Keras.

---

# Important Note About Pretrained Models

VGG16, ResNet50, and EfficientNetB0 use:

```python
weights="imagenet"
```

Therefore, TensorFlow may need to download the pretrained ImageNet weights the first time the notebook is executed.

An internet connection may be required during the initial model setup.

---

# Results

The notebook produces several outputs for comparing the four architectures:

### Numerical Comparison

- Test Accuracy
- Test Loss
- Training Time

### Training Curves

- Validation Accuracy Comparison
- Validation Loss Comparison

### Final Comparison

- Test Accuracy Bar Chart
- Best Performing Model

The exact numerical results should be taken from the output generated during the notebook's execution.

---

# Limitations

The project uses only:

```text
10,000 training images
2,000 test images
```

instead of the complete CIFAR-10 dataset.

This was done to make the multi-model comparison more manageable, but it means the results may differ from models trained on the complete dataset.

The models are also trained for only:

```text
5 epochs
```

which is relatively short for a comprehensive deep learning experiment.

Additionally, VGG16, ResNet50, and EfficientNetB0 use frozen pretrained feature extractors, so the comparison is not equivalent to training all four architectures entirely from scratch.

---

# Future Improvements

The project can be extended by:

- Training on the complete CIFAR-10 dataset
- Increasing the number of epochs
- Fine-tuning the pretrained VGG16, ResNet50, and EfficientNetB0 layers
- Comparing different learning rates
- Comparing different batch sizes
- Adding Early Stopping
- Adding learning-rate scheduling
- Using additional data augmentation
- Tracking precision, recall, and F1-score
- Generating confusion matrices for each model
- Analyzing incorrect predictions
- Comparing parameter counts
- Comparing inference time
- Comparing memory requirements
- Saving the best-performing model
- Deploying the selected model as an image classification application

---

# Key Takeaway

This project provides a practical comparison of four well-known CNN architectures on the same image classification problem.

The experiment follows:

```text
Same Dataset
      ↓
Same General Training Setup
      ↓
Different CNN Architectures
      ↓
Train
      ↓
Evaluate
      ↓
Compare
```

The project demonstrates that model selection should not depend solely on accuracy.

A useful comparison should also consider:

```text
Accuracy
Loss
Training Time
Generalization
Model Complexity
```

By comparing **AlexNet, VGG16, ResNet50, and EfficientNetB0**, this project provides practical experience with both **CNN architecture design** and **transfer learning using pretrained models**.

---

# Author

**Lavanya**

Machine Learning | Deep Learning | Computer Vision | Neural Networks
