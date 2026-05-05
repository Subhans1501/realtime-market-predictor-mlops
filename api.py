from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tf_keras.models import load_model

app = FastAPI(title="Market Movement API")

print("Loading GRU Model...")
model = load_model("models/gru_model.keras")
print("Model loaded successfully!")

class MarketInput(BaseModel):
    close_price: float
    sentiment_score: float

@app.post("/predict")
def predict_market(data: MarketInput):

    input_sequence = np.array([[[data.close_price, data.sentiment_score]] * 5])

    prob = model.predict(input_sequence)[0][0]
    
    direction = "Upward Trend 🟢" if prob > 0.5 else "Downward Trend 🔴"
    confidence = float(prob if prob > 0.5 else (1 - prob)) * 100
    
    return {
        "prediction": direction, 
        "confidence": round(confidence, 2)
    }