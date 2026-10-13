# Multi-purpose production Dockerfile for Orange Telecom Churn Prediction System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ ./src/
COPY dataset/ ./dataset/
COPY models/ ./models/
COPY orange.png app.py orange_churn.py pyproject.toml ./

# Install local package in editable mode
RUN pip install --no-cache-dir -e .

# Environment settings
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

# Default command starts FastAPI service (can be overridden to Streamlit in compose)
CMD ["uvicorn", "telecom_churn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
