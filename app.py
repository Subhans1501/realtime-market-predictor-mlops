import streamlit as st
import requests
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
st.markdown("<p style='text-align: center; color: gray;'>Powered by FastAPI Backend & GRU Model</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Inputs")
    dummy_close = st.number_input("Recent Close Price ($)", value=150.00, step=0.50, format="%.2f")
    
with col2:
    st.subheader("Sentiment Data")
    sentiment_map = {"Positive (Bullish)": 1.0, "Neutral": 0.0, "Negative (Bearish)": -1.0}
    selected_sentiment = st.selectbox("Recent News Sentiment", list(sentiment_map.keys()))
    dummy_sentiment = sentiment_map[selected_sentiment]

st.write("") 

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Predict Next Day Direction")

if predict_clicked:
    with st.spinner('Querying FastAPI Inference Engine...'):
        time.sleep(1)
        
        try:
            api_url = "http://127.0.0.1:8000/predict"
            payload = {
                "close_price": dummy_close,
                "sentiment_score": dummy_sentiment
            }
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("Model Inference Results")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Predicted Market Direction", value=result["prediction"])
                with res_col2:
                    st.metric(label="Model Confidence", value=f"{result['confidence']}%")
                    
                st.success("Inference completed successfully via REST API!")
            else:
                st.error(f"API Error: {response.status_code}. Ensure FastAPI is running.")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure you run `uvicorn api:app --reload` in another terminal!")