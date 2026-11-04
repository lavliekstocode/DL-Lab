# Audio Classification, Speech-to-Text and Text-to-Speech

A deep learning project that combines **environmental sound classification, speech recognition, and speech synthesis** using Python. The project uses the ESC-50 dataset to classify environmental sounds with a PyTorch Multi-Layer Perceptron (MLP), extracts MFCC audio features, converts speech into text using Whisper Tiny, and generates spoken output using Google Text-to-Speech (gTTS).

## Project Overview

This project integrates three audio-processing tasks into a single pipeline:

1. **Audio Classification:** Classifies environmental sounds into 50 categories using a PyTorch-based MLP.
2. **Speech-to-Text (STT):** Converts spoken audio into text using OpenAI Whisper Tiny.
3. **Text-to-Speech (TTS):** Converts text into spoken audio using Google Text-to-Speech.

The ESC-50 dataset is used for environmental sound classification, while a separate speech recording is used for speech recognition.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Building and training the MLP |
| Librosa | Audio loading and MFCC feature extraction |
| NumPy | Numerical computations |
| Pandas | Dataset handling and evaluation results |
| Scikit-learn | Data preprocessing and performance evaluation |
| Matplotlib | Training visualizations |
| Seaborn | Confusion matrix visualization |
| Hugging Face Transformers | Whisper Tiny speech recognition |
| gTTS | Text-to-speech conversion |
| Joblib | Saving preprocessing objects |

## Dataset: ESC-50

The project uses the **ESC-50 (Environmental Sound Classification) dataset**, a publicly available collection of environmental audio recordings.

- 2,000 environmental audio recordings.
- 50 sound categories.
- 40 recordings per category.
- Each recording is approximately 5 seconds long.
- Audio files are provided in WAV format.

Examples of sound categories include:

- Dog
- Rain
- Thunderstorm
- Airplane
- Clock tick
- Door knock
- Car horn
- Wind

Dataset repository: [ESC-50 GitHub Repository](https://github.com/karolpiczak/ESC-50)

The dataset is automatically downloaded and extracted during the first execution of the notebook.

## Project Workflow

### 1. Dataset Download and Loading
- Downloads the ESC-50 dataset from GitHub.
- Extracts the audio files and metadata.
- Loads the CSV file containing audio filenames and their corresponding categories.

### 2. Dataset Selection
- Uses up to 1,000 recordings by default to reduce training time.
- Selects approximately equal numbers of recordings from each sound category.
- The complete dataset can be used by setting `MAX_SAMPLES = None`.

### 3. Audio Feature Extraction using MFCC

**Mel-Frequency Cepstral Coefficients (MFCC)** are extracted from each audio recording to represent its frequency characteristics.

The audio processing steps include:
- Loading audio at a sampling rate of 22,050 Hz.
- Converting audio into mono format.
- Extracting 40 MFCC coefficients.
- Calculating the mean and standard deviation of each coefficient.

This produces an **80-dimensional feature vector** for every audio recording.

### 4. Data Preprocessing
- Encodes sound categories into numerical labels using LabelEncoder.
- Splits the dataset into 80% training data and 20% testing data.
- Uses stratified sampling to maintain class proportions.
- Applies StandardScaler to normalize the extracted features.

The scaler is fitted only on the training data to avoid data leakage.

### 5. Deep Learning Model: PyTorch MLP

A Multi-Layer Perceptron is implemented using PyTorch.

**Model Architecture:**

| Layer | Configuration |
|---|---|
| Input Layer | 80 MFCC features |
| Hidden Layer 1 | 256 neurons, ReLU |
| Dropout | 0.30 |
| Hidden Layer 2 | 128 neurons, ReLU |
| Dropout | 0.20 |
| Output Layer | 50 sound categories |

**Training Configuration:**

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 15
- Random Seed: 42

The model automatically uses CUDA if a compatible GPU is available; otherwise, it runs on the CPU.

### 6. Model Evaluation

The trained model is evaluated on the testing dataset using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-Score
- Classification Report

Macro averaging gives equal importance to each sound category.

### 7. Data Visualization

The project generates visualizations to understand model performance:

- Training loss across epochs.
- Confusion matrix showing actual and predicted sound categories.
- Classification report containing precision, recall, and F1-score for individual categories.

### 8. Custom Audio Prediction

The trained model can classify a new environmental audio recording.

To test a custom recording, set:

```python
AUDIO_PATH = "my_sound.wav"
```

The program:
1. Loads the audio file.
2. Extracts its MFCC features.
3. Applies the previously fitted scaler.
4. Passes the features to the trained MLP.
5. Displays the predicted sound category and model confidence.
6. Plays the audio recording.

If no custom recording is provided, the notebook demonstrates prediction using a held-out test recording.

### 9. Speech-to-Text using Whisper Tiny

The project includes an optional speech recognition module using the `openai/whisper-tiny` model.

- Accepts a separate speech recording named `speech.wav`.
- Uses Hugging Face Transformers for automatic speech recognition.
- Generates a text transcript from the spoken audio.

To use a different recording, modify:

```python
SPEECH_PATH = "speech.wav"
```

**Note:** ESC-50 contains environmental sounds, not speech transcription data. Whisper operates on a separate speech recording and is not evaluated using the ESC-50 dataset.

### 10. Text-to-Speech using gTTS

Google Text-to-Speech converts text into spoken audio.

The default mode speaks the predicted environmental sound category.

Two modes are available:

| Mode | Function |
|---|---|
| `label` | Speaks the predicted environmental sound category |
| `transcript` | Speaks the Whisper-generated transcript |

Change the mode using:

```python
TTS_MODE = "label"
```

The generated audio is saved as:

```text
tts_output.mp3
```

The notebook also plays the generated speech.

An internet connection is required for gTTS.

## Installation and Execution

### Requirements

- Python 3.9 or later recommended.
- Google Colab or local Jupyter Notebook.
- Internet connection for dataset download, Whisper model download, and gTTS.
- GPU is optional.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Step 2: Install Dependencies

The notebook automatically installs the required libraries in its first cell.

Alternatively, install them manually:

```bash
pip install numpy pandas matplotlib seaborn librosa soundfile scikit-learn torch transformers gTTS joblib requests
```

### Step 3: Run the Notebook

Open the notebook in Jupyter Notebook or Google Colab and execute the cells sequentially.

The first execution downloads the ESC-50 dataset automatically.

### Step 4: Test Custom Audio

Set the audio file path in the notebook:

```python
AUDIO_PATH = "my_sound.wav"
```

For speech recognition, provide a separate recording:

```python
SPEECH_PATH = "speech.wav"
```

## Project Structure

```text
Audio-Classification-STT-TTS/
│
├── Audio_Classification_STT_TTS.ipynb
│
├── ESC-50-master/
│   ├── audio/
│   └── meta/
│       └── esc50.csv
│
├── esc50_audio_classifier.pt
├── esc50_scaler.joblib
├── esc50_label_encoder.joblib
├── audio_classification_metrics.csv
│
├── tts_output.mp3
│
└── README.md
```

*Generated files and the dataset directory will appear after running the notebook. They do not need to be present in the repository beforehand.*

## Saved Model and Files

After execution, the project saves:

| File | Description |
|---|---|
| `esc50_audio_classifier.pt` | Trained PyTorch model weights and model configuration |
| `esc50_scaler.joblib` | Fitted feature scaler |
| `esc50_label_encoder.joblib` | Label encoder containing sound categories |
| `audio_classification_metrics.csv` | Evaluation metrics |
| `tts_output.mp3` | Generated speech audio |

The model checkpoint, scaler, and label encoder can be reused for future predictions.

## Important Notes

- The default configuration uses 1,000 recordings rather than the complete ESC-50 dataset to reduce execution time.
- Setting `MAX_SAMPLES = None` enables the complete dataset.
- The model uses MFCC statistical features rather than raw audio waveforms.
- Whisper Tiny is an optional speech recognition component and requires a separate speech recording.
- gTTS requires an active internet connection.
- Model confidence represents the highest predicted softmax probability; it should not be interpreted as a guarantee of prediction correctness.
- Actual evaluation metrics are generated during execution and should be reported using the notebook's output rather than assumed values.

## Future Improvements

- Experiment with CNN and RNN architectures for audio classification.
- Explore spectrogram-based audio classification.
- Improve performance through hyperparameter tuning.
- Experiment with larger Whisper models for speech recognition.
- Develop a web interface for real-time audio classification.
- Integrate all three modules into a unified audio-processing application.

## Author

**Lavanya**



---

*This project demonstrates the integration of deep learning, audio feature extraction, speech recognition, and speech synthesis in a unified Python-based application.*
