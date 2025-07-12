FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Use minimal requirements first
COPY requirements-minimal.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p chroma_db logs

CMD uvicorn real_estate_rag_system:app --host 0.0.0.0 --port $PORT
