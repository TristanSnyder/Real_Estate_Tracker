FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create directories
RUN mkdir -p chroma_db logs

# Railway requires this format for PORT
EXPOSE $PORT

# Health check (optional but helpful)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Start command - make sure it binds to 0.0.0.0
CMD uvicorn real_estate_rag_system:app --host 0.0.0.0 --port $PORT --workers 1
