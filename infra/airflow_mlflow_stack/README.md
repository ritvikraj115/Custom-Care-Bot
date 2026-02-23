# All-in-One Docker Stack (VM: Azure or EC2)

This folder now contains one production stack for a single Linux VM:

- `frontend` (React + Nginx)
- `backend` (Node/Express)
- `python-doc-service` (FastAPI)
- `postgres` (Airflow + MLflow metadata)
- `mlflow`
- `airflow-apiserver`
- `airflow-scheduler`
- `airflow-dag-processor`

## 1. Prepare `.env`

```bash
cd infra/airflow_mlflow_stack
cp .env.example .env
```

Set required values in `.env`:
- `MONGO_URI`
- `JWT_SECRET`
- `GEMINI_API_KEY`
- `ES_URLS`
- `ES_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AIRFLOW_FERNET_KEY` (required)
- `AIRFLOW_ADMIN_PASSWORD` (change default)

Generate fernet key (once):

```bash
docker run --rm apache/airflow:3.1.7 python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

## 2. Start Everything

```bash
docker compose up -d --build
```

## 3. Access Services

- Frontend: `http://<vm-public-ip>:80`
- Backend API: `http://<vm-public-ip>:5000`
- Python docs API: `http://<vm-public-ip>:8000/docs`
- Airflow API/UI: `http://<vm-public-ip>:8091`
- MLflow: `http://<vm-public-ip>:5001`

## 4. Smoke Test

Run after startup:

```bash
chmod +x smoke-test.sh
./smoke-test.sh
```

## 5. Stop / Restart

```bash
docker compose down
docker compose up -d
```

Reset all persistent data:

```bash
docker compose down -v
```
