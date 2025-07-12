FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db /app/logs

# Expose port (Railway will set PORT env var)
EXPOSE $PORT

# Start command (Railway compatible)
CMD python -m uvicorn real_estate_rag_system:app --host 0.0.0.0 --port $PORT
