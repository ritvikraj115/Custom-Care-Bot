# Single-VM Docker Deployment (Azure VM or AWS EC2)

This repo is simplified for one deployment method:
- Run everything with Docker Compose from `infra/airflow_mlflow_stack`.

Services included:
- Frontend
- Backend
- Python doc service
- Airflow API + scheduler + dag-processor
- MLflow
- Postgres (for Airflow + MLflow metadata)

## 1. Create Linux VM

Use Ubuntu 22.04 LTS on either:
- Azure Virtual Machine
- AWS EC2 instance

Recommended minimum:
- 4 vCPU
- 16 GB RAM
- 80+ GB disk

Open inbound ports:
- `22` (SSH)
- `80` (frontend)
- `5000` (backend optional direct access)
- `8000` (python doc service optional direct access)
- `8091` (Airflow)
- `5001` (MLflow)

## 2. Install Docker + Compose Plugin

Run on VM:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 3. Clone Repo

```bash
git clone <your-repo-url> adv_project
cd adv_project
```

## 4. Configure Environment

```bash
cd infra/airflow_mlflow_stack
cp .env.example .env
```

Edit `.env` and fill required values:
- `MONGO_URI`
- `JWT_SECRET`
- `GEMINI_API_KEY`
- `ES_URLS`
- `ES_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AIRFLOW_FERNET_KEY`
- `AIRFLOW_ADMIN_PASSWORD`

Generate `AIRFLOW_FERNET_KEY`:

```bash
docker run --rm apache/airflow:3.1.7 python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

## 5. Deploy All Services

From `infra/airflow_mlflow_stack`:

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs -f --tail=150
```

## 6. Verify Endpoints

- Frontend: `http://<VM_PUBLIC_IP>:80`
- Backend: `http://<VM_PUBLIC_IP>:5000`
- Python API docs: `http://<VM_PUBLIC_IP>:8000/docs`
- Airflow: `http://<VM_PUBLIC_IP>:8091`
- MLflow: `http://<VM_PUBLIC_IP>:5001`

Run stack smoke test:

```bash
chmod +x smoke-test.sh
./smoke-test.sh
```

## 7. DVC One-Time Setup in Python Container

Run once:

```bash
docker compose exec python-doc-service python -m dvc remote remove localstore || true
docker compose exec python-doc-service python -m dvc remote add -d prod s3://dvc-customcare/adv_project/dvc || true
docker compose exec python-doc-service python -m dvc remote modify prod region us-west-2
docker compose exec python-doc-service python -m dvc push
```

## 8. Operations

Restart one service:

```bash
docker compose restart python-doc-service
```

Update code and redeploy:

```bash
git pull
docker compose up -d --build
```

Stop stack:

```bash
docker compose down
```

Delete all volumes/data:

```bash
docker compose down -v
```
