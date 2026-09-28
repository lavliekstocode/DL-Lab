# BERT-Tiny Based News Text Classification

## 📌 Project Overview

This project implements a lightweight Transformer-based text classification system using **BERT-tiny** and the **AG News dataset**.

The model reads a news article and predicts which of four categories it belongs to:

- World
- Sports
- Business
- Sci/Tech

A smaller BERT model is used instead of standard BERT so that the project can run on systems with limited RAM, GPU and disk space.

---

## 🎯 Objectives

- Understand how Transformer-based text classification works.
- Apply tokenization to real-world news text.
- Fine-tune a pretrained BERT-based model for classification.
- Evaluate the model using standard classification metrics.
- Visualize classification performance using a confusion matrix.
- Use the trained model to classify new unseen news articles.

---

## 📊 Dataset

The project uses the **AG News dataset**.

The dataset contains news articles belonging to four categories:

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Dataset source:

Hugging Face: https://huggingface.co/datasets/fancyzhx/ag_news

The complete dataset is larger, but this implementation uses a smaller subset to keep training lightweight.

### Dataset used in this project

- Training samples: 7,200
- Validation samples: 800
- Test samples: 2,000
- Number of classes: 4

---

## 🧠 Model

The project uses:

**BERT-tiny**

Model:

`prajjwal1/bert-tiny`

BERT-tiny is a compact pretrained Transformer model. It follows the same general Transformer-based approach as BERT but has a much smaller architecture.

This makes it suitable for experimentation on laptops and systems with limited computational resources.

### Model Pipeline

News Article  
↓  
BERT Tokenizer  
↓  
Token IDs + Attention Mask  
↓  
BERT-tiny  
↓  
Classification Layer  
↓  
4 Output Classes  
↓  
Predicted News Category

---

## ⚙️ Methodology

### 1. Dataset Loading

The AG News dataset is loaded using the Hugging Face `datasets` library.

### 2. Data Sampling

A smaller subset of the original dataset is selected to reduce training time and memory consumption.

### 3. Tokenization

The news articles are converted into tokens using the BERT tokenizer.

The maximum sequence length is limited to **128 tokens**.

### 4. Dynamic Padding

Instead of padding every article to the maximum length, padding is performed dynamically for each batch.

This reduces unnecessary memory usage.

### 5. Model Fine-Tuning

The pretrained BERT-tiny model is fine-tuned for four-class news classification.

Training configuration:

- Learning Rate: `2e-5`
- Batch Size: `8`
- Epochs: `5`
- Optimizer: AdamW through Hugging Face Trainer
- Weight Decay: `0.01`
- Maximum Sequence Length: `128`

### 6. Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

### 7. New Text Prediction

A prediction function is included to classify a new news article and display the predicted category along with the model's confidence.

---

## 📦 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

---

## 📁 Project Structure

    BERT-News-Classification/
    │
    ├── bert_news_classification.py
    ├── README.md
    └── bert_tiny_agnews_results/
        └── model checkpoints

---

## 🔧 Installation

Install the required libraries using:

    pip install transformers datasets scikit-learn torch accelerate matplotlib seaborn

---

## ▶️ How to Run

### 1. Clone the repository

    git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Open the project folder

    cd BERT-News-Classification

### 3. Install dependencies

    pip install transformers datasets scikit-learn torch accelerate matplotlib seaborn

### 4. Run the Python program

    python bert_news_classification.py

The program will:

1. Load AG News.
2. Prepare the training, validation and test sets.
3. Load BERT-tiny.
4. Tokenize the news articles.
5. Train the model for 5 epochs.
6. Calculate evaluation metrics.
7. Generate a classification report.
8. Display a confusion matrix.
9. Test the model on a new news article.

---

## 📈 Evaluation Metrics

### Accuracy

Accuracy represents the percentage of news articles that were classified correctly.

    Accuracy = Correct Predictions / Total Predictions

### Precision

Precision measures how many of the articles predicted as a particular category actually belonged to that category.

### Recall

Recall measures how many of the actual articles belonging to a category were correctly identified.

### F1 Score

F1 Score combines precision and recall into a single metric.

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## 📊 Results

Run the model to generate the actual test metrics.

The program automatically prints:

    Accuracy
    Precision
    Recall
    F1 Score

A detailed classification report is also generated for:

    World
    Sports
    Business
    Sci/Tech

The confusion matrix shows which categories the model classifies correctly and which categories it tends to confuse.

---

## 🔍 Example Prediction

The project includes an example news article related to artificial intelligence and computing:

    Apple announced a new generation of artificial intelligence
    chips designed to improve the performance of its latest
    computing devices and machine learning applications.

The model returns:

    Predicted Category: Sci/Tech

    Confidence: XX.XX%

The exact confidence value depends on the trained model.

---

## 🧩 Why BERT-Tiny?

Standard BERT models can require significant memory and storage.

BERT-tiny was selected because:

- It is much smaller.
- It requires fewer computational resources.
- Training is faster.
- It is easier to run on a personal computer.
- It still demonstrates pretrained Transformer-based text classification.

This makes it a practical choice for an academic NLP project.

---

## 🚀 Future Improvements

The project can be improved by:

- Training on the complete AG News dataset.
- Increasing the maximum sequence length.
- Using a larger pretrained Transformer model.
- Hyperparameter tuning.
- Increasing the number of training epochs.
- Adding learning-rate scheduling.
- Comparing BERT-tiny with DistilBERT and BERT-base.
- Deploying the classifier as a web application.
- Adding a user interface for real-time news classification.

---

## 📚 References

- AG News Dataset:
  https://huggingface.co/datasets/fancyzhx/ag_news

- Hugging Face Transformers:
  https://huggingface.co/docs/transformers

- BERT:
  https://arxiv.org/abs/1810.04805

- BERT-tiny:
  https://huggingface.co/prajjwal1/bert-tiny

---

## 👩‍💻 Project Summary

This project demonstrates how a pretrained Transformer model can be fine-tuned for multi-class text classification.

Using BERT-tiny and the AG News dataset, the system learns patterns in news articles and classifies them into World, Sports, Business, or Sci/Tech categories.

The lightweight implementation makes the project suitable for running on systems with limited computational resources while still demonstrating the main concepts involved in Transformer-based NLP classification.
