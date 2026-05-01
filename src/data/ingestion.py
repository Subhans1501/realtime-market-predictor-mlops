import yfinance as yf
import pandas as pd
import numpy as np
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from datetime import datetime, timedelta

def fetch_financial_data(ticker="AAPL", period="2y"):
    print(f"Attempting to fetch data for {ticker} from Yahoo Finance...")
    try:

        df = yf.download(ticker, period=period, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            print("Successfully fetched live data from Yahoo Finance!")
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Yahoo API Error: {e}")

    print("\n[WARNING] Yahoo Finance blocked the request or network failed.")
    print("[INFO] Generating synthetic market data to keep the MLOps pipeline running...\n")
    
    dates = [datetime.today() - timedelta(days=x) for x in range(500)]
    dates.reverse() # Oldest to newest

    close_prices = np.linspace(150, 190, 500) + np.random.normal(0, 3, 500)
    
    df_dummy = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.normal(0, 1, 500),
        'High': close_prices + np.abs(np.random.normal(1, 1, 500)),
        'Low': close_prices - np.abs(np.random.normal(1, 1, 500)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 500)
    })
    return df_dummy

def simulate_news_sentiment(dates):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    dummy_headlines = [
        "Tech stocks rally as market shows strong recovery.",
        "Inflation fears trigger massive sell-offs globally.",
        "Company maintains steady growth despite market conditions.",
        "Federal reserve signals potential rate cuts."
    ]
    
    for _ in dates:
        headline = random.choice(dummy_headlines)
        score = analyzer.polarity_scores(headline)
        
        if score['compound'] >= 0.05:
            sentiments.append(1)
        elif score['compound'] <= -0.05:
            sentiments.append(-1)
        else:
            sentiments.append(0)
            
    return sentiments

def main():
    os.makedirs("data/raw", exist_ok=True)
    df = fetch_financial_data()
    
    print("Applying sentiment analysis...")
    df['Sentiment_Score'] = simulate_news_sentiment(df['Date'])

    df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True) 
    output_path = "data/raw/market_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Pipeline complete. Dataset saved to {output_path} with {len(df)} rows.")
if __name__ == "__main__":
    main()
