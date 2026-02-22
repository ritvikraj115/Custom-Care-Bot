# Airflow + MLflow Docker Stack (Isolated)

This is a separate stack from `airflow_local/` and can run independently.

Services:
- `postgres` (shared by Airflow + MLflow)
- `airflow-apiserver` (Airflow API/UI on `:8091` by default)
- `airflow-scheduler`
- `airflow-dag-processor`
- `mlflow` (MLflow tracking server on `:5001` by default)

Airflow image stays lightweight:
- uses official `apache/airflow:3.1.7`
- no custom heavy ML dependencies are installed in Airflow

## 1. Configure

```powershell
cd infra/airflow_mlflow_stack
Copy-Item .env.example .env
```

Edit `.env`:
- Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
- Keep `MLFLOW_S3_BUCKET=mlflow-custom-care`.
- Keep `DVC_S3_BUCKET=dvc-customcare` (used by your app-side DVC commands).
- Set Airflow admin credentials.
- Optional: set `AIRFLOW_ML_SERVICE_URL` to your python doc service URL.

## 2. Start

```powershell
docker compose up -d --build
```

## 3. Access

- Airflow API/UI: `http://localhost:8081`
- Airflow API/UI: `http://localhost:8091`
- MLflow: `http://localhost:5001`

## 4. Wire Into Your App

Set these in `python_doc_service` runtime environment:
- `AIRFLOW_AUTOTRIGGER_ENABLED=true`
- `AIRFLOW_API_BASE_URL=http://localhost:8091`
- `AIRFLOW_API_USERNAME=<airflow-admin-username>`
- `AIRFLOW_API_PASSWORD=<airflow-admin-password>`
- `MLFLOW_ENABLED=true`
- `MLFLOW_TRACKING_URI=http://localhost:5001`
- `ES_URLS=https://my-elasticsearch-project-ab7db2.es.us-central1.gcp.elastic.cloud:443`
- `ES_API_KEY=<your elastic api key>`

## 5. DVC Bucket Setup Command

Run in `python_doc_service`:

```powershell
python -m dvc remote remove localstore
python -m dvc remote add -d prod s3://dvc-customcare/adv_project/dvc
python -m dvc remote modify prod region us-west-2
python -m dvc remote modify --local prod access_key_id <AWS_ACCESS_KEY_ID>
python -m dvc remote modify --local prod secret_access_key <AWS_SECRET_ACCESS_KEY>
python -m dvc push
```

## 6. Stop

```powershell
docker compose down
```

To reset postgres data:

```powershell
docker compose down -v
```
