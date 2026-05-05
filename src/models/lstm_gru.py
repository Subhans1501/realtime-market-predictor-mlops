import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df, look_back=5):

    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[['Close', 'Sentiment_Score']])
    target = df['Target_Direction'].values
    
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back)])
        y.append(target[i + look_back])
        
    return np.array(X), np.array(y)

def train_and_log_model(model_type, X_train, y_train, X_test, y_test, input_shape):

    print(f"\nTraining {model_type} Model...")

    mlflow.set_experiment("Market_Movement_Prediction")
    
    with mlflow.start_run(run_name=f"{model_type}_Architecture"):
        epochs = 10
        batch_size = 16

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("look_back_window", 5)

        model = Sequential()
        if model_type == "LSTM":
            model.add(LSTM(32, activation='relu', input_shape=input_shape))
        elif model_type == "GRU":
            model.add(GRU(32, activation='relu', input_shape=input_shape))
            
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        predictions = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("models", exist_ok=True)
        model.save(f"models/{model_type.lower()}_model.keras")
        print(f"\n{model_type} Results -> Accuracy: {acc:.4f} | F1: {f1:.4f}")

def main():
    print("Loading data for advanced model training...")
    df = pd.read_csv("data/raw/market_data.csv")
    
    X, y = prepare_data(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    input_shape = (X_train.shape[1], X_train.shape[2])
    train_and_log_model("LSTM", X_train, y_train, X_test, y_test, input_shape)
    train_and_log_model("GRU", X_train, y_train, X_test, y_test, input_shape)
    print("\nTraining complete! Run 'mlflow ui' to view your dashboard.")
if __name__ == "__main__":
    main()