# LSTM-Based Traffic Volume Forecasting 🚗

## 📌 Project Overview

This project develops an LSTM (Long Short-Term Memory) based time-series forecasting model to predict hourly traffic volume.

The project uses the Metro Interstate Traffic Volume Dataset, which contains historical traffic observations along with date and time information. The LSTM model learns patterns from previous traffic observations and predicts the traffic volume for the next hour.

This project demonstrates how deep learning can be applied to smart-city transportation management and traffic forecasting.

---

## 🎯 Objectives

- Analyze historical traffic volume data.
- Preprocess and normalize time-series data.
- Create sequential input data for an LSTM model.
- Train an LSTM neural network for traffic forecasting.
- Predict future hourly traffic volume.
- Evaluate the model using MAE, RMSE, and MAPE.
- Visualize actual vs predicted traffic volume.

---

## 📊 Dataset

### Metro Interstate Traffic Volume Dataset

The dataset contains hourly traffic volume measurements recorded on an interstate highway.

Important columns include:

- `date_time` – Date and time of the observation
- `traffic_volume` – Number of vehicles recorded during the observation period
- `weather_main` – Main weather condition
- `weather_description` – Detailed weather condition
- `temp` – Temperature
- `rain_1h` – Rainfall during the previous hour
- `snow_1h` – Snowfall during the previous hour
- `clouds_all` – Cloud coverage
- `holiday` – Holiday information

For this implementation, `traffic_volume` is used as the primary time-series variable.

---

## 🧠 Methodology

The forecasting pipeline follows these steps:

**Historical Traffic Data → Data Cleaning → Date-Time Conversion → Chronological Sorting → Traffic Volume Extraction → Min-Max Normalization → 80% Training / 20% Testing → Create 24-Hour Sequences → LSTM Model → Traffic Prediction → Inverse Scaling → Model Evaluation → Visualization**

### 1. Data Preprocessing

The dataset is loaded using Pandas. The `date_time` column is converted into a datetime format and the observations are sorted chronologically.

Duplicate timestamps and missing traffic-volume values are handled before training.

### 2. Data Normalization

Traffic volume is scaled to a range between 0 and 1 using `MinMaxScaler`.

This helps the neural network train more efficiently and prevents large numerical values from dominating the learning process.

### 3. Train-Test Split

The data is divided chronologically:

- **80% → Training data**
- **20% → Testing data**

Random shuffling is avoided because time-series data must preserve its temporal order.

### 4. Sequence Creation

The model uses the previous **24 hours** of traffic observations to predict the traffic volume for the next hour.

For example:

**Previous 24 Hours → LSTM → Next Hour Traffic Volume**

This allows the model to learn short-term temporal traffic patterns.

### 5. Model Training

An LSTM neural network is trained using the generated sequences. Early stopping is used to prevent unnecessary training and reduce overfitting.

### 6. Prediction

After training, the model predicts traffic volume for the test dataset. The scaled predictions are converted back to the original traffic-volume scale.

### 7. Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## 🤖 LSTM Architecture

The model consists of:

**Input: Previous 24 Hours**

↓

**LSTM Layer – 64 Units**

↓

**Dropout – 20%**

↓

**LSTM Layer – 32 Units**

↓

**Dropout – 20%**

↓

**Dense Layer – 16 Neurons, ReLU**

↓

**Output Layer – 1 Neuron**

↓

**Predicted Traffic Volume**

### Why LSTM?

LSTM networks are designed for sequential and time-dependent data.

Traffic volume is not independent from one hour to another. Traffic during a particular hour can be influenced by patterns observed during previous hours.

LSTM networks can retain relevant information from previous time steps through their internal memory and gating mechanisms, making them suitable for traffic forecasting.

---

## ⚙️ Technologies Used

- **Python**
- **Pandas** – Data loading and preprocessing
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Scikit-learn** – Data scaling and evaluation metrics
- **TensorFlow / Keras** – LSTM model development
- **Jupyter Notebook** – Development environment

---

## 📁 Project Structure

    LSTM-Traffic-Forecasting/
    │
    ├── Metro_Interstate_Traffic_Volume.csv
    ├── traffic_forecasting.ipynb
    ├── README.md
    └── requirements.txt

---

## 📦 Installation

Clone the repository:

    git clone <your-github-repository-url>
    cd LSTM-Traffic-Forecasting

Install the required libraries:

    pip install numpy pandas matplotlib scikit-learn tensorflow

---

## ▶️ How to Run

1. Download the Metro Interstate Traffic Volume dataset.
2. Place the CSV file in the project directory.
3. Open the Jupyter Notebook.
4. Make sure the dataset filename matches:

   `Metro_Interstate_Traffic_Volume.csv`

5. Run all cells.
6. The program will:
   - Load the dataset
   - Clean the data
   - Sort observations chronologically
   - Normalize traffic volume
   - Create 24-hour sequences
   - Train the LSTM model
   - Generate predictions
   - Calculate evaluation metrics
   - Display visualizations
   - Forecast the next hour's traffic volume

---

## 📈 Evaluation Metrics

The model uses three evaluation metrics.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted traffic volume.

**MAE = Average(|Actual - Predicted|)**

A lower MAE indicates that predictions are closer to the actual traffic volume.

### Root Mean Squared Error (RMSE)

RMSE gives greater importance to larger prediction errors.

**RMSE = √(Average((Actual - Predicted)²))**

A lower RMSE indicates fewer large prediction errors.

### Mean Absolute Percentage Error (MAPE)

MAPE expresses the prediction error as a percentage.

**MAPE = Average(|Actual - Predicted| / Actual) × 100**

Zero-valued observations are excluded from the MAPE calculation because division by zero makes MAPE undefined.

---

## 📊 Results

The model produces three main evaluation metrics:

- **MAE:** Average traffic-volume prediction error
- **RMSE:** Error measure that penalizes larger prediction errors
- **MAPE:** Average percentage prediction error

The exact metric values may vary depending on the dataset version, preprocessing, TensorFlow version, hardware, and model training.

The project also generates an **Actual vs Predicted Traffic Volume** graph to visually evaluate forecasting performance.

---

## 📉 Visualizations

The project generates the following visualizations:

### 1. Historical Traffic Volume

Shows traffic volume across the complete dataset and helps identify general traffic patterns.

### 2. Training vs Validation Loss

Shows how the LSTM model learns during training and helps identify potential overfitting.

### 3. Actual vs Predicted Traffic

Compares actual traffic volume against traffic volume predicted by the LSTM model.

### 4. Next-Hour Forecast

Displays the predicted traffic volume for the next hour.

---

## 🌆 Real-World Applications

Traffic forecasting can support:

- Smart-city transportation systems
- Traffic congestion monitoring
- Route planning
- Intelligent transportation systems
- Traffic signal optimization
- Infrastructure planning
- Emergency response planning
- Public transportation management

Accurate traffic forecasting can help transportation authorities understand expected traffic conditions and make better-informed transportation decisions.

---

## 🔮 Future Improvements

The current implementation primarily uses historical traffic volume.

The model can be extended into a **multivariate LSTM** by incorporating additional variables such as:

- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather conditions
- Holidays
- Day of the week
- Hour of the day

This would allow the model to learn relationships between traffic volume and external factors.

Other possible improvements include:

- Bidirectional LSTM
- GRU networks
- CNN-LSTM
- Attention mechanisms
- Transformer-based forecasting
- Hyperparameter optimization
- Longer forecasting horizons
- Multi-step traffic forecasting

---

## 🚀 Future Scope

A more advanced version of the project could provide traffic forecasts for multiple future hours instead of only the next hour.

The system could also be integrated with a dashboard to display:

**Current Traffic → Predicted Traffic → Congestion Level → Traffic Alert → Suggested Route**

This could make the project more suitable for a real-world smart transportation platform.

---

## 📌 Project Highlights

- Uses **Deep Learning** for time-series forecasting.
- Implements an **LSTM neural network**.
- Uses historical traffic observations as sequential input.
- Preserves chronological ordering during training and testing.
- Uses a **24-hour lookback window**.
- Evaluates predictions using MAE, RMSE, and MAPE.
- Provides visual comparison between actual and predicted traffic.
- Includes next-hour traffic forecasting.
- Has potential applications in **smart cities and intelligent transportation systems**.

---

## 👩‍💻 Project Information

**Project Type:** Machine Learning / Deep Learning

**Domain:** Smart Cities & Intelligent Transportation Systems

**Task:** Time-Series Forecasting

**Model:** Long Short-Term Memory (LSTM)

**Dataset:** Metro Interstate Traffic Volume

**Prediction Target:** Hourly Traffic Volume

---

## ✅ Conclusion

This project demonstrates how an LSTM-based deep learning model can be used for hourly traffic volume forecasting.

By analyzing the previous 24 hours of traffic observations, the model learns temporal patterns and generates predictions for future traffic volume.

The project highlights the application of time-series deep learning to a real-world transportation problem and demonstrates how predictive models can contribute to smarter and more efficient transportation management systems.

---

## 📚 Dataset Reference

**Dataset:** Metro Interstate Traffic Volume

The dataset is available through the UCI Machine Learning Repository and other public machine-learning repositories.

**Dataset Name:** Metro Interstate Traffic Volume

**Task:** Traffic Volume Time-Series Forecasting
