FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

EXPOSE 5000

CMD ["bash", "-lc", "MLFLOW_DB_HOST=${MLFLOW_DB_HOST:-postgres}; MLFLOW_DB_PORT=${MLFLOW_DB_PORT:-5432}; mlflow server --host 0.0.0.0 --port ${PORT:-5000} --backend-store-uri postgresql+psycopg2://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_HOST}:${MLFLOW_DB_PORT}/${MLFLOW_DB_NAME} --artifacts-destination s3://${MLFLOW_S3_BUCKET}/artifacts --serve-artifacts"]
