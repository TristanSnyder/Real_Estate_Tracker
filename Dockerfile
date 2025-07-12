FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install with verbose output
COPY requirements.txt .
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Verify ChromaDB installation
RUN python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"

# Copy application
COPY . .
RUN mkdir -p chroma_db logs

# Start command
CMD sh -c "uvicorn real_estate_rag_system:app --host 0.0.0.0 --port ${PORT:-8000}"
