import streamlit as st
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Market Predictor MLOps", 
    page_icon="📈", 
    layout="centered"
)

st.title("Real-Time Market Movement Predictor")
st.write("Predict the next day's stock market movement using a Deep Learning GRU model.")

API_URL = "http://127.0.0.1:8000/predict"

tab1, tab2 = st.tabs(["⚡ Live Auto-Fetch", "✍️ Manual Entry"])

with tab1:
    st.header("Live Market Data Auto-Fetch")
    st.write("Automatically pull the latest Apple (AAPL) price and analyze live news headlines.")
    
    if st.button("Fetch Live Data & Predict", type="primary"):
        with st.spinner("Fetching live data from Wall Street..."):
            try:

                aapl = yf.Ticker("AAPL")
                live_price = aapl.history(period="1d")['Close'].iloc[-1]

                news_data = aapl.news
                headlines = [article['title'] for article in news_data[:5]]

                analyzer = SentimentIntensityAnalyzer()
                scores = [analyzer.polarity_scores(title)['compound'] for title in headlines]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                if avg_score > 0.05:
                    sentiment_text = "Positive (Bullish)"
                elif avg_score < -0.05:
                    sentiment_text = "Negative (Bearish)"
                else:
                    sentiment_text = "Neutral"

                st.info(f"**Live AAPL Price:** ${live_price:.2f}")
                st.write("**Top Headlines Analyzed:**")
                for title in headlines[:3]:
                    st.caption(f"- {title}")
                st.info(f"**Calculated Sentiment:** {sentiment_text} (VADER Score: {avg_score:.2f})")

                payload = {"close_price": float(live_price), "sentiment": sentiment_text}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.success(f"**AI Model Prediction for Tomorrow:** {prediction}")
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")

with tab2:
    st.header("Manual Data Entry")
    st.write("Test specific historical scenarios (e.g., Apple's Q2 Earnings breakout).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        manual_price = st.number_input("Recent Close Price ($)", min_value=0.0, value=284.18, step=0.5)
    with col2:
        manual_sentiment = st.selectbox(
            "Recent News Sentiment", 
            ["Negative (Bearish)", "Neutral", "Positive (Bullish)"], 
            index=2
        )
        
    if st.button("Predict Next Day Direction"):
        with st.spinner("Analyzing sequence..."):
            try:
                payload = {"close_price": manual_price, "sentiment": manual_sentiment}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.success(f"**AI Model Prediction for Tomorrow:** {prediction}")
                else:
                    st.error("Error connecting to the GRU Model backend.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is FastAPI running?")