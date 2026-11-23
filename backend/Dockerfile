FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV FASTEMBED_CACHE_PATH=/opt/docuai/models
RUN python -c 'from fastembed import TextEmbedding; TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir="/opt/docuai/models", threads=1)'

COPY . .

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app /opt/docuai
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
