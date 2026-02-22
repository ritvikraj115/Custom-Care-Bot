# Airflow Local Setup (PowerShell)

This folder contains an official Docker Compose based Airflow setup, wired to:
- DAGs in `../python_doc_service/airflow/dags`
- project code in `../python_doc_service`
- lightweight orchestration (no TensorFlow install in Airflow container)
- autocomplete retrain trigger via ML service API (`/autocomplete/train`)
- hdbscan retrain trigger via ML service API (`/hdbscan/train`)

## Configure
1. Edit `.env` if you want defaults:
   - `AIRFLOW_ML_CLIENT_ID`
   - `AIRFLOW_ML_BOT_ID`
   If left blank, pass both dynamically in `dag_run.conf` at trigger time.
2. Optionally set:
   - `AIRFLOW_WEB_PORT` (default in this repo: `8081`)
   - `AIRFLOW_ML_SERVICE_URL` (default `http://host.docker.internal:8000`)
   - `AIRFLOW_HDBSCAN_REBUILD_MODE`
   - `AIRFLOW_HDBSCAN_PDF_MANIFEST`
   - `AIRFLOW_DEPLOY_CMD`

## Start
```powershell
cd airflow_local
.\setup-airflow.ps1
```

## Stop
```powershell
docker compose down
```

## Trigger DAGs
```powershell
docker compose exec airflow-scheduler airflow dags unpause hdbscan_pipeline
docker compose exec airflow-scheduler airflow dags unpause autocomplete_pipeline
docker compose exec airflow-scheduler airflow dags trigger hdbscan_pipeline --conf "{\"client_id\":\"<CLIENT_ID>\",\"bot_id\":\"<BOT_ID>\",\"rebuild_mode\":\"full\",\"pdf_manifest\":\"data/dvc/pdf_manifest.json\"}"
docker compose exec airflow-scheduler airflow dags trigger autocomplete_pipeline --conf "{\"client_id\":\"<CLIENT_ID>\",\"bot_id\":\"<BOT_ID>\"}"
```

## Notes
- `autocomplete_pipeline` now calls your ML service endpoint and relies on existing in-service logic for:
  - retrain condition checks
  - MLflow tracking
  - DVC `autocomplete_track` trigger
- `hdbscan_pipeline` now calls your ML service endpoint and keeps DVC execution in backend service flow (`/process` or `/hdbscan/train`), so Airflow stays lightweight.
