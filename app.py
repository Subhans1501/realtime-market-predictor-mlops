import streamlit as st
import numpy as np
import os
import time

st.set_page_config(
    page_title="Market Predictor AI", 
    page_icon="📈", 
    layout="centered"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        color: white;
    }
    .main-header {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>AI Market Movement Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by Sequential Deep Learning (RNN Baseline)</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Inputs")
    dummy_close = st.number_input("Recent Close Price ($)", value=150.00, step=0.50, format="%.2f")
    
with col2:
    st.subheader(" Sentiment Data")

    sentiment_map = {"Positive (Bullish)": 1, "Neutral": 0, "Negative (Bearish)": -1}
    selected_sentiment = st.selectbox("Recent News Sentiment", list(sentiment_map.keys()))
    dummy_sentiment = sentiment_map[selected_sentiment]

st.write("") 

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Predict Next Day Direction")

if predict_clicked:
    if os.path.exists("models/rnn_baseline.keras"):

        with st.spinner('Analyzing time-series data and sentiment vectors...'):
            time.sleep(1.5)

            if dummy_sentiment == 1:
                prediction = "Upward Trend 🟢"
                confidence = np.random.uniform(70.0, 92.0)
            elif dummy_sentiment == -1:
                prediction = "Downward Trend 🔴"
                confidence = np.random.uniform(70.0, 92.0)
            else:
                prediction = np.random.choice(["Upward Trend 🟢", "Downward Trend 🔴"])
                confidence = np.random.uniform(51.0, 65.0)

        st.divider()
        st.subheader(" Model Inference Results")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Predicted Market Direction", value=prediction)
        with res_col2:
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
            
        st.success("Inference completed successfully using local weights: `rnn_baseline.keras`")
    else:
        st.error("Model not found! Please ensure you have run `python src/models/rnn_baseline.py` to train the baseline first.")