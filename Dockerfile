FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn[standard]
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn real_estate_rag_system:app --host 0.0.0.0 --port ${PORT:-8000}"]
