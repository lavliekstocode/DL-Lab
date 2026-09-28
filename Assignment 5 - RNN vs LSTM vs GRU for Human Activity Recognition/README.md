# RNN vs LSTM vs GRU for Human Activity Recognition 📱

## 📌 Project Overview

This project implements and compares three recurrent neural network architectures — **Simple RNN, LSTM, and GRU** — for human activity recognition using smartphone sensor data.

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**, where sensor readings collected from a smartphone are used to classify different physical activities.

The three models are trained using the same dataset and evaluated using **Accuracy, Precision, Recall, and F1 Score**.

The goal is to understand how different recurrent architectures perform when working with sequential sensor data.

---

## 🎯 Objectives

- Implement a Simple RNN for sequence classification.
- Implement an LSTM model for sequence classification.
- Implement a GRU model for sequence classification.
- Train all three models using the same dataset.
- Compare their classification performance.
- Evaluate the models using Accuracy, Precision, Recall, and F1 Score.
- Analyze the models using confusion matrices.
- Compare training and validation performance using graphs.

---

## 📊 Dataset

### UCI Human Activity Recognition Using Smartphones

The dataset contains sensor measurements collected from smartphones worn by participants while performing different physical activities.

The smartphone sensors include:

- Body acceleration
- Total acceleration
- Body gyroscope

The data is organized into sequences of **128 time steps**.

Each time step contains measurements from multiple sensor signals.

### Activities

The model classifies six different activities:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

---

## 🧠 Methodology

The overall workflow is:

**Smartphone Sensor Data → Data Loading → Sensor Signal Combination → Sequence Formation → RNN / LSTM / GRU → Activity Prediction → Evaluation → Model Comparison**

### 1. Data Loading

The UCI HAR dataset is downloaded and extracted automatically if it is not already present.

The training and testing sensor signals are then loaded using NumPy.

### 2. Sensor Data

Nine sensor signals are used:

- Body acceleration X
- Body acceleration Y
- Body acceleration Z
- Body gyroscope X
- Body gyroscope Y
- Body gyroscope Z
- Total acceleration X
- Total acceleration Y
- Total acceleration Z

Each sample contains:

- **128 time steps**
- **9 sensor features**

Therefore, the input to the recurrent networks has the shape:

**Samples × 128 Time Steps × 9 Features**

### 3. Activity Labels

The original activity labels range from 1 to 6.

They are converted to zero-based labels from 0 to 5 for model training.

### 4. Model Training

Three different recurrent architectures are trained:

- Simple RNN
- LSTM
- GRU

All three models use the same input data, optimizer, batch size, validation split, and maximum number of epochs to make the comparison fair.

The maximum number of training epochs is **10**.

Early stopping is also used to stop training if the validation loss stops improving.

---

## 🤖 Models Used

### Simple RNN

The Simple RNN is the basic recurrent architecture used as the baseline model.

It processes the sensor sequence one time step at a time and maintains information from previous time steps.

### LSTM

Long Short-Term Memory (LSTM) is designed to handle longer-term dependencies in sequential data.

It uses memory cells and gates to control which information should be retained or discarded.

### GRU

Gated Recurrent Unit (GRU) is another gated recurrent architecture.

It is simpler than LSTM while still being able to capture dependencies across time steps.

---

## 🏗️ Model Architecture

All three models use the same general structure, with only the recurrent layer changing.

**Input**

128 time steps × 9 sensor features

↓

**Recurrent Layer**

RNN / LSTM / GRU — 64 units

↓

**Dropout**

30%

↓

**Dense Layer**

32 neurons with ReLU activation

↓

**Output Layer**

6 neurons with Softmax activation

↓

**Predicted Activity**

---

## ⚙️ Technologies Used

- **Python**
- **NumPy** – Numerical operations and sensor data processing
- **Pandas** – Data handling
- **Matplotlib** – Visualization
- **Scikit-learn** – Evaluation metrics
- **TensorFlow / Keras** – Deep learning models
- **Jupyter Notebook** – Development environment

---

## 📁 Project Structure

    Human-Activity-Recognition/
    │
    ├── UCI HAR Dataset/
    ├── human_activity_recognition.ipynb
    ├── README.md
    └── requirements.txt

---

## 📦 Installation

Clone the repository:

    git clone <your-github-repository-url>
    cd Human-Activity-Recognition

Install the required libraries:

    pip install numpy pandas matplotlib scikit-learn tensorflow

---

## ▶️ How to Run

1. Open the Jupyter Notebook.
2. Run the notebook cells in order.
3. The program automatically downloads the UCI HAR dataset if it is not already available.
4. The sensor data is loaded and combined.
5. The three models are created.
6. RNN, LSTM, and GRU models are trained for a maximum of 10 epochs.
7. Each model is evaluated on the test dataset.
8. Performance comparison graphs and confusion matrices are generated.

---

## 📈 Evaluation Metrics

The models are evaluated using four main classification metrics.

### Accuracy

Accuracy measures the percentage of activity sequences classified correctly.

**Accuracy = Correct Predictions / Total Predictions**

### Precision

Precision measures how many of the samples predicted as a particular activity actually belong to that activity.

Higher precision means fewer false positive predictions.

### Recall

Recall measures how many of the actual samples belonging to an activity were correctly identified.

Higher recall means fewer activities are missed by the model.

### F1 Score

F1 Score combines precision and recall into a single metric.

It is particularly useful when comparing classification performance across multiple classes.

---

## 📊 Model Comparison

The notebook produces a comparison table containing:

- Accuracy
- Precision
- Recall
- F1 Score

The models are compared using the same testing dataset.

The comparison helps identify how the three recurrent architectures behave when processing sequential smartphone sensor data.

---

## 📉 Visualizations

The project generates several visualizations.

### 1. Validation Accuracy

The validation accuracy of RNN, LSTM, and GRU is plotted across the training epochs.

This helps compare how quickly each architecture learns.

### 2. Validation Loss

Validation loss is plotted for all three models to observe how well each model generalizes during training.

### 3. Performance Comparison

A bar chart compares the Accuracy, Precision, Recall, and F1 Score of the three models.

### 4. Confusion Matrices

A confusion matrix is generated for each model.

The matrix shows which activities are correctly classified and which activities are confused with one another.

---

## 🔍 Why Compare RNN, LSTM and GRU?

All three architectures are designed to process sequential information, but they handle temporal dependencies differently.

### RNN

Simple architecture and relatively lightweight, but it can have difficulty retaining information over longer sequences.

### LSTM

Uses additional gates and memory cells to preserve useful information for longer periods.

### GRU

Uses a simpler gating structure than LSTM and can provide competitive performance with fewer parameters.

Comparing them on the same dataset allows their classification performance to be studied under the same conditions.

---

## 📊 Results

The three models were trained and evaluated on the same test dataset using a maximum of 10 epochs.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **RNN** | **69.97%** | **71.39%** | **69.97%** | **69.86%** |
| LSTM | 50.08% | 45.95% | 50.08% | 45.91% |
| GRU | 53.92% | 55.02% | 53.92% | 48.78% |

The Simple RNN achieved the highest overall performance in this experiment, with an accuracy of **69.97%** and an F1 Score of **69.86%**.

The GRU achieved an accuracy of **53.92%**, while the LSTM achieved **50.08%**.

These results are specific to the selected architecture, hyperparameters, dataset preprocessing, and 10-epoch training limit. Therefore, the results should not be interpreted as indicating that Simple RNN is generally superior to LSTM or GRU. With additional training, hyperparameter tuning, and architecture optimization, the performance of the models may change.

## 🌍 Real-World Applications

Human activity recognition can be used in:

- Smartphone fitness applications
- Wearable devices
- Fitness tracking
- Healthcare monitoring
- Elderly activity monitoring
- Smart home systems
- Rehabilitation systems
- Sports analytics
- Context-aware mobile applications

---

## 🔮 Future Improvements

The current project uses nine sensor signals. It can be extended by incorporating additional features and architectures.

Possible improvements include:

- Adding more sensor features
- Hyperparameter tuning
- Increasing the sequence length
- Bidirectional LSTM
- Bidirectional GRU
- CNN-LSTM
- Attention mechanisms
- Transformer-based sequence classification
- Real-time activity recognition
- Deployment on mobile or wearable devices

---

## 🚀 Future Scope

A real-time version of the system could receive sensor readings directly from a smartphone or wearable device and classify the user's activity continuously.

The system could follow:

**Live Sensor Data → Sequence Buffer → Trained Model → Activity Prediction → Real-Time Display**

This could enable applications such as automatic workout tracking, health monitoring, fall detection, and activity-aware mobile applications.

---

## 📌 Project Highlights

- Implements three recurrent architectures.
- Uses real-world smartphone sensor data.
- Performs multiclass sequence classification.
- Compares Simple RNN, LSTM, and GRU under the same conditions.
- Uses a maximum of 10 training epochs.
- Uses early stopping to avoid unnecessary training.
- Evaluates models using Accuracy, Precision, Recall, and F1 Score.
- Generates confusion matrices for detailed analysis.
- Includes validation accuracy and loss comparisons.
- Demonstrates a practical application of deep learning to human activity recognition.

---

## 👩‍💻 Project Information

**Project Type:** Deep Learning / Sequence Classification

**Domain:** Human Activity Recognition

**Dataset:** UCI Human Activity Recognition Using Smartphones

**Models:** Simple RNN, LSTM, GRU

**Input:** Smartphone Sensor Time-Series Data

**Number of Classes:** 6

**Sequence Length:** 128 Time Steps

**Sensor Features:** 9

**Maximum Epochs:** 10

---

## ✅ Conclusion

This project demonstrates and compares three recurrent neural network architectures — RNN, LSTM, and GRU — for classifying human activities from smartphone sensor data.

The models process sequences of sensor measurements and classify them into six different physical activities.

By evaluating Accuracy, Precision, Recall, F1 Score, validation performance, and confusion matrices, the project provides a practical comparison of different recurrent architectures for sequence classification.

The experiment demonstrates how recurrent neural networks can be applied to real-world sensor-based applications such as fitness tracking, healthcare monitoring, wearable technology, and smart mobile systems.

---

## 📚 Dataset Reference

**UCI Human Activity Recognition Using Smartphones Dataset**

Dataset developed by researchers from the University of Genova and available through the UCI Machine Learning Repository.

**Dataset:** Human Activity Recognition Using Smartphones

**Task:** Multiclass Sequence Classification

**Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
