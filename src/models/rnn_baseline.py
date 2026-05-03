import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

def prepare_data(df, look_back=5):
    scaler = MinMaxScaler()
    # Using Close price and Sentiment as features
    features = scaler.fit_transform(df[['Close', 'Sentiment_Score']])
    target = df['Target_Direction'].values
    
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back)])
        y.append(target[i + look_back])
        
    return np.array(X), np.array(y)

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/raw/market_data.csv")
    
    print("Preparing sequence data...")
    X, y = prepare_data(df)
    
    # Simple Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Building RNN Baseline Model...")
    model = Sequential([
        SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training Model...")
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/rnn_baseline.keras")
    print("Baseline RNN trained and saved to models/rnn_baseline.keras")

if __name__ == "__main__":
    main()