FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python packages in stages (lighter first)
RUN pip install --no-cache-dir --timeout 300 \
    fastapi uvicorn[standard] pydantic sqlalchemy psycopg2-binary redis python-dotenv

# Install ML packages separately (these are the heavy ones)
RUN pip install --no-cache-dir --timeout 600 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 600 \
    transformers sentence-transformers chromadb

# Install remaining packages
RUN pip install --no-cache-dir --timeout 300 \
    scrapy pandas requests beautifulsoup4 celery feedparser yfinance langchain

# Copy application code
COPY . .

# Create directories
RUN mkdir -p chroma_db logs

# Start command
CMD uvicorn real_estate_rag_system:app --host 0.0.0.0 --port $PORT
