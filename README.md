# Real-Time Market Movement Predictor

An end-to-end Machine Learning Operations (MLOps) pipeline that predicts the next day's stock market movement (Upward/Downward) based on recent close prices and news sentiment.

This project features a deep learning model trained using Kaggle resources, served via a FastAPI backend, visualized with a Streamlit frontend, and fully containerized for deployment on an AWS server.

---

## Features

- **Deep Learning Engine:** Utilizes a Gated Recurrent Unit (GRU) model with a 5-day look-back window to capture market trends and sequential momentum.

- **RESTful Backend:** FastAPI handles incoming predictions, processes raw data through a `MinMaxScaler`, and serves model inferences.

- **Interactive Frontend:** Streamlit provides a clean, user-friendly UI to input price and sentiment data and view the prediction.

- **Robust MLOps Deployment:** Fully containerized using Docker, avoiding dependency conflicts (utilizing the stable `.h5` model format and `tf-keras`), and deployed on an AWS EC2 instance.

---

## Tech Stack

### Machine Learning
- TensorFlow
- Keras (`tf-keras`)
- Scikit-Learn (`MinMaxScaler`)
- Pandas
- NumPy
- MLflow

### Backend API
- FastAPI
- Uvicorn

### Frontend UI
- Streamlit
- Requests

### DevOps / Deployment
- Docker
- Git/GitHub
- AWS (Ubuntu EC2)

---

## Project Structure

```text
├── app.py                 # Streamlit frontend UI
├── api.py                 # FastAPI backend server
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration for AWS deployment
├── models/
│   └── gru_model.h5       # Trained GRU model (Legacy HDF5 format for stability)
├── src/
│   └── models/
│       └── lstm_gru.py    # Model training, MLflow tracking, and data prep script
└── README.md
```

---

## Model Architecture & Pipeline

### Input
Recent Close Price ($) and News Sentiment (Negative / Neutral / Positive).

### Preprocessing
Sentiment is mapped to a numerical score. Both price and sentiment are scaled to a 0–1 range using `MinMaxScaler`.

### Sequencing
The system appends the latest inputs to a 5-day historical look-back sequence to provide context.

### Prediction
The sequence is fed into the GRU network.

### Output
A sigmoid activation layer calculates the probability.

- Score > 0.5 → **UPWARD (Bullish)**
- Score < 0.5 → **DOWNWARD (Bearish)**

---

## Local Setup & Development

To run this project locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/subhans1501/realtime-market-predictor-mlops.git
cd realtime-market-predictor-mlops
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Start the FastAPI Backend (Terminal 1)

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start the Streamlit Frontend (Terminal 2)

```bash
streamlit run app.py
```

---

## Docker & AWS Deployment

To deploy the production-ready container to an AWS Ubuntu server:

### 1. Pull Latest Code

```bash
git pull origin main
```

### 2. Clean Old Docker Cache (Optional)

```bash
sudo docker system prune -a --volumes
```

### 3. Build Docker Image

```bash
sudo docker build -t market-predictor .
```

### 4. Run Docker Container

```bash
sudo docker run -d -p 8000:8000 -p 8501:8501 market-predictor
```

### 5. Access the Application

#### Streamlit UI
```text
http://<your-aws-ip>:8501
```

#### FastAPI Docs
```text
http://<your-aws-ip>:8000/docs
```

---

## Retraining the Model

If you wish to retrain the model with new data or update the architecture:

```bash
python src/data/lstm_gru.py
```

This script will:
- Train the model
- Track metrics using MLflow
- Automatically save the best-performing model to the `models/` folder as `.h5`

---