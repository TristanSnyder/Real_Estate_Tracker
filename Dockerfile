FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p chroma_db logs

# Use Python to start the app (handles PORT properly)
CMD ["python", "-m", "real_estate_rag_system"]
