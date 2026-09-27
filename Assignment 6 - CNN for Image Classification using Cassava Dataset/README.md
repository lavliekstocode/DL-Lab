# Cassava Leaf Disease Classification using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify cassava leaf images into different disease categories.

The project demonstrates an end-to-end computer vision workflow, including dataset loading, image preprocessing, data augmentation, CNN architecture design, model training, validation, performance evaluation, disease prediction, and model saving.

---

## Project Overview

Cassava is an important crop, and identifying diseases from leaf images can help in understanding crop health.

This project trains a CNN to learn visual patterns from cassava leaf images and classify them into the disease classes present in the dataset.

The images are resized to:

```text
128 × 128 pixels
```

and processed as RGB images.

After training, the model can also be used to predict the disease category of a new cassava leaf image and provide a confidence score.

---

## Machine Learning Pipeline

```text
Cassava Leaf Image Dataset
          ↓
Load Images from Folders
          ↓
80% Training / 20% Validation
          ↓
Resize Images to 128 × 128
          ↓
Data Augmentation
          ↓
Pixel Rescaling
          ↓
Convolutional Neural Network
          ↓
Model Training
          ↓
Validation Evaluation
          ↓
Predictions
          ↓
Confusion Matrix + Classification Report
          ↓
New Leaf Image Prediction
          ↓
Save Trained Model
```

---

## Dataset

The project loads the image dataset from:

```text
./data
```

TensorFlow's `image_dataset_from_directory()` automatically assigns labels based on the folder names.

The dataset is divided into:

- **80% training data**
- **20% validation data**

A fixed random seed of `42` is used to make the train-validation split reproducible.

---

## Image Preprocessing

### Image Resizing

All images are resized to:

```text
128 × 128 × 3
```

where:

- `128` = image height
- `128` = image width
- `3` = RGB color channels

Using a fixed image size ensures that every image has the same dimensions before being passed to the CNN.

### Pixel Rescaling

The model uses:

```python
layers.Rescaling(1./255)
```

to convert pixel values from:

```text
0 – 255
```

to:

```text
0 – 1
```

This normalizes the input values before they are processed by the network.

---

## Data Augmentation

Data augmentation is applied to the training images to create slightly modified versions of the original images.

The project uses:

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.1)
layers.RandomZoom(0.1)
```

### Augmentation Techniques

| Technique | Purpose |
|------------|------------|
| Random Flip | Creates horizontally flipped image variations |
| Random Rotation | Slightly changes the orientation of images |
| Random Zoom | Creates images at slightly different scales |

Data augmentation provides the model with greater image variety and can help reduce overfitting.

---

## Dataset Pipeline Optimization

The TensorFlow dataset pipeline is optimized using:

```python
tf.data.AUTOTUNE
```

along with:

```python
cache()
shuffle()
prefetch()
```

### Cache

Caching keeps the dataset available after the first pass through the data.

### Shuffle

Shuffling changes the order of training samples to help prevent the model from learning patterns based on the original ordering of the dataset.

### Prefetch

Prefetching prepares upcoming batches while the model is processing the current batch.

This helps make the training pipeline more efficient.

---

## CNN Architecture

The project uses a **Sequential Convolutional Neural Network** consisting of four convolution blocks followed by fully connected classification layers.

```text
Input Image
128 × 128 × 3
       ↓
Data Augmentation
       ↓
Rescaling
       ↓
Conv2D — 32 Filters
       ↓
MaxPooling2D
       ↓
Conv2D — 64 Filters
       ↓
MaxPooling2D
       ↓
Conv2D — 128 Filters
       ↓
MaxPooling2D
       ↓
Conv2D — 256 Filters
       ↓
MaxPooling2D
       ↓
Flatten
       ↓
Dense — 128 Neurons
       ↓
Dropout — 50%
       ↓
Dense — Number of Classes
       ↓
Softmax
```

The number of convolution filters progressively increases:

```text
32 → 64 → 128 → 256
```

This allows deeper layers to learn increasingly complex visual features from the leaf images.

---

## Convolutional Layers

### Convolution Block 1

The first convolution layer uses:

```text
32 filters
3 × 3 kernel
ReLU activation
```

The initial layer learns basic visual features such as:

- Edges
- Lines
- Simple textures

---

### Convolution Block 2

The second convolution layer uses:

```text
64 filters
3 × 3 kernel
ReLU activation
```

With more filters, the network can learn more detailed patterns from the images.

---

### Convolution Block 3

The third convolution layer uses:

```text
128 filters
3 × 3 kernel
ReLU activation
```

At this stage, the network can learn more complex shapes and textures.

---

### Convolution Block 4

The fourth convolution layer uses:

```text
256 filters
3 × 3 kernel
ReLU activation
```

The deeper layer extracts high-level visual features that are useful for distinguishing between disease classes.

---

## Max Pooling

Each convolution block is followed by:

```python
layers.MaxPooling2D((2, 2))
```

Max pooling reduces the spatial dimensions of the feature maps while retaining important features.

This reduces the amount of computation required by later layers and helps the network focus on the most significant features.

---

## Flatten Layer

After the convolution blocks, the feature maps are passed through:

```python
layers.Flatten()
```

The Flatten layer converts the two-dimensional feature maps into a single vector.

This allows the extracted visual features to be passed into the fully connected Dense layers.

---

## Dense Layer

The classification section contains:

```python
layers.Dense(
    128,
    activation="relu"
)
```

This layer combines the features extracted by the convolutional layers before the final disease classification.

---

## Dropout

The model uses:

```python
layers.Dropout(0.5)
```

A dropout rate of `0.5` means that 50% of the neurons are randomly disabled during training.

This helps reduce overfitting by preventing the network from becoming overly dependent on specific neurons.

---

## Output Layer

The final layer is:

```python
layers.Dense(
    num_classes,
    activation="softmax"
)
```

The number of output neurons is determined dynamically from the number of disease classes:

```python
num_classes = len(class_names)
```

Softmax converts the output into probabilities for each disease class.

The class with the highest probability is selected as the model's prediction.

---

## Model Compilation

The CNN is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Optimizer

**Adam** is used to update the model's weights during training.

Adam adapts the learning rate during optimization.

### Loss Function

**Sparse Categorical Cross-Entropy** is used because the disease labels are represented as integer class values.

### Evaluation Metric

**Accuracy** measures how many images are classified correctly.

---

## Model Training

The CNN is trained for:

```text
20 epochs
```

Training is performed using:

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
```

During training:

1. Images are passed through the CNN.
2. The model generates predictions.
3. The loss between predictions and actual labels is calculated.
4. Backpropagation updates the model's weights.
5. The process is repeated across multiple epochs.
6. Validation performance is measured after each epoch.

---

## Training and Validation Accuracy

The project plots both training and validation accuracy:

```python
history.history["accuracy"]
history.history["val_accuracy"]
```

The resulting curves help understand how the model's accuracy changes during training.

Comparing training and validation accuracy can also provide an indication of possible overfitting.

---

## Training and Validation Loss

The project also plots:

```python
history.history["loss"]
history.history["val_loss"]
```

Loss represents how far the model's predictions are from the correct labels.

Ideally, the loss should decrease as the model learns.

Comparing training and validation loss can help identify whether the model is generalizing well or beginning to overfit.

---

## Model Evaluation

After training, the model is evaluated on the complete validation dataset:

```python
loss, accuracy = model.evaluate(val_ds)
```

The notebook reports:

```text
Validation Loss
Validation Accuracy
Validation Accuracy (%)
```

This provides a quantitative measure of the model's performance on images that were not used for training.

---

## Generating Predictions

The trained model generates predictions for validation images.

For each batch:

```python
predictions = model.predict(images)
```

The model returns probability values for each class.

The predicted class is obtained using:

```python
predicted_classes = np.argmax(
    predictions,
    axis=1
)
```

`argmax` selects the class with the highest predicted probability.

---

## Confusion Matrix

A confusion matrix is generated using:

```python
cm = confusion_matrix(
    y_true,
    y_pred
)
```

The confusion matrix shows how images from each actual disease class are classified by the model.

It can help identify:

- Correct classifications
- Incorrect classifications
- Disease classes that are frequently confused with one another

This provides more detailed insight than accuracy alone.

---

## Classification Report

The project generates a classification report using:

```python
classification_report(
    y_true,
    y_pred,
    target_names=class_names
)
```

The report provides performance metrics for each disease class, including:

- Precision
- Recall
- F1-score
- Support

This makes it possible to analyze the model's performance class by class.

---

## Predicting a New Leaf Image

The trained CNN is also tested on a separate image:

```text
test_leaf.jpeg
```

The image is loaded using:

```python
load_img(
    image_path,
    target_size=IMG_SIZE
)
```

This ensures that the new image has the same dimensions as the images used during training.

---

## Preparing the New Image

The loaded image is converted into a NumPy array:

```python
img_array = img_to_array(img)
```

A batch dimension is then added:

```python
img_array = tf.expand_dims(
    img_array,
    axis=0
)
```

The resulting input has the expected structure:

```text
(batch, height, width, channels)
```

---

## Disease Prediction

The trained model predicts the disease using:

```python
prediction = model.predict(img_array)
```

The predicted class is selected using:

```python
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
```

The model also calculates the confidence of the prediction:

```python
confidence = prediction[0][predicted_index] * 100
```

The final output displays:

```text
Predicted Disease: <class>
Confidence: <percentage>%
```

---

## Prediction Visualization

The test image is displayed together with the model's prediction and confidence score.

Example output format:

```text
Prediction: <Predicted Disease>
Confidence: XX.XX%
```

This provides a simple visual way to inspect the model's prediction for an individual leaf image.

---

## Model Saving

The trained CNN is saved in Keras format using:

```python
model.save(
    "cassava_leaf_disease_cnn.keras"
)
```

The saved model can be loaded later without retraining the network.

This makes it possible to reuse the trained CNN in future applications or deployment workflows.

---

## Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Convolutional Neural Networks**
- **Computer Vision**
- **Deep Learning**

---

## Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Convolutional Neural Networks
- Image Classification
- Image Preprocessing
- Image Resizing
- Pixel Normalization
- Data Augmentation
- Convolutional Layers
- Feature Extraction
- Max Pooling
- Flattening
- Dense Layers
- ReLU Activation
- Softmax Activation
- Dropout
- Forward Propagation
- Backpropagation
- Adam Optimization
- Sparse Categorical Cross-Entropy
- Model Training
- Validation
- Confusion Matrix
- Classification Reports
- Precision
- Recall
- F1-score
- Prediction Confidence
- Model Saving

---

## Project Structure

```text
Cassava-Leaf-Disease-CNN/
│
├── data/
│   ├── disease_class_1/
│   ├── disease_class_2/
│   ├── ...
│   └── disease_class_n/
│
├── CNN.ipynb
│
├── test_leaf.jpeg
│
├── cassava_leaf_disease_cnn.keras
│
└── README.md
```

The exact disease class folder names depend on the dataset used with the notebook.

---

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Cassava-Leaf-Disease-CNN
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 3. Prepare the Dataset

Place the cassava leaf image dataset inside:

```text
./data
```

Each disease class should have its own folder so that TensorFlow can automatically assign labels based on the directory structure.

### 4. Add a Test Image

Place a separate leaf image in the project directory with the filename:

```text
test_leaf.jpeg
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the CNN notebook and run the cells sequentially.

---

## Results

The project evaluates the CNN using the validation dataset and reports:

- Validation loss
- Validation accuracy
- Precision
- Recall
- F1-score
- Class-wise support

It also generates:

- Sample training images
- Training vs. validation accuracy graph
- Training vs. validation loss graph
- Confusion matrix
- Classification report
- New leaf disease prediction
- Prediction confidence

The actual numerical performance depends on the dataset and training run.

---

## Limitations

The model's performance depends heavily on:

- Dataset quality
- Number of images per disease class
- Image diversity
- Lighting conditions
- Background variations
- Quality of disease labels

The project uses a validation split rather than a separate test dataset for final evaluation.

Therefore, validation accuracy should not automatically be interpreted as a definitive real-world performance measure.

The model also should not be treated as a replacement for expert agricultural or plant-pathology diagnosis.

---

## Future Improvements

The project can be extended by:

- Using a separate test dataset
- Increasing the size and diversity of the training dataset
- Handling class imbalance
- Adding more advanced augmentation techniques
- Experimenting with different CNN architectures
- Using Transfer Learning
- Comparing models such as MobileNet, ResNet, or EfficientNet
- Adding Batch Normalization
- Performing hyperparameter tuning
- Using learning-rate scheduling
- Adding early stopping
- Analyzing incorrectly classified images
- Deploying the model as a web or mobile application
- Building a real-time leaf disease detection system

---

## Key Takeaway

This project demonstrates how a **Convolutional Neural Network can be used to classify plant leaf images based on learned visual features**.

The complete workflow covers:

**Load → Preprocess → Augment → Extract Features → Train → Validate → Evaluate → Predict → Save**

The project provides practical experience with the fundamental components of computer vision and deep learning, while also demonstrating how model evaluation techniques such as confusion matrices and classification reports can be used to understand classification performance beyond simple accuracy.

---

## Author

**Lavanya**

Machine Learning | Deep Learning | Computer Vision | Neural Networks
