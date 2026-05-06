FROM python:3.10-slim

WORKDIR /app

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for Streamlit frontend (and API if applicable)
EXPOSE 8000 8501

# Boot up the Frontend (Adjust CMD if you don't have an api.py yet)
CMD ["sh", "-c", "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]