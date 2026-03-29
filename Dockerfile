FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY context_models.py .
COPY query_normalization.py .
COPY query_grounding.py .
COPY query_routing.py .
COPY start.py .

CMD ["python", "start.py"]
