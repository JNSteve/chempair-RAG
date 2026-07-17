FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY citation_extraction.py .
COPY context_models.py .
COPY kb_info.py .
COPY query_normalization.py .
COPY session_store.py .
COPY upstream_errors.py .
COPY start.py .
COPY corpus/manifest.yaml corpus/manifest.yaml

CMD ["python", "start.py"]
