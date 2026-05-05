import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from tf_keras.models import Sequential
from tf_keras.layers import SimpleRNN, Dense

def prepare_data(df, look_back=5):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[['Close', 'Sentiment_Score']])
    target = df['Target_Direction'].values
    
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back)])
        y.append(target[i + look_back])
        
    return np.array(X), np.array(y)

def main():
    mlflow.set_experiment("Market_Movement_Prediction")
    
    print("Loading dataset...")
    df = pd.read_csv("data/raw/market_data.csv")
    
    look_back = 5
    epochs = 10
    
    print("Preparing sequence data...")
    X, y = prepare_data(df, look_back=look_back)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    with mlflow.start_run(run_name="RNN_Baseline"):
        print("Building RNN Baseline Model...")
        model = Sequential([
            SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Training Model...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=1)
        
        print("Evaluating Model...")
        predictions = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("look_back_window", look_back)
        mlflow.log_param("model_type", "RNN")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/rnn_baseline.keras")
        print(f"Baseline RNN trained. Accuracy: {acc:.4f} | F1: {f1:.4f}")

main()