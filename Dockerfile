FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN mkdir -p chroma_db logs

# Fix the PORT issue - use shell form instead of exec form
CMD uvicorn real_estate_rag_system:app --host 0.0.0.0 --port ${PORT:-8000}
