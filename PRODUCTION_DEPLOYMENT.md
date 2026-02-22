# Render-Only Production Rollout (Exact Order)

This runbook deploys everything on Render:
- Frontend (`chatbot`)
- Node backend (`server`)
- Python doc service (`python_doc_service`)
- MLflow (private service)
- Airflow API + scheduler + dag-processor (private/worker services)
- Two Render Postgres databases (Airflow metadata, MLflow backend)

No secrets are committed to git. All sensitive values are `sync: false` in `render.yaml`.

## 1. Minimum Prerequisites

1. Render account with access to your repo.
2. MongoDB URI (unchanged, from your current setup).
3. Elasticsearch endpoint + API key.
4. S3 buckets:
- `dvc-customcare`
- `mlflow-custom-care`
5. AWS IAM credentials with S3 permissions for those buckets.

## 2. Repo Safety Before Deploy

1. Keep real secrets only in local `.env` files or Render dashboard.
2. Do not commit `.env` files.
3. This repo now includes root `.gitignore` rules that ignore `.env` and `.env.*` (except `.env.example`).

## 3. Exact Rollout Order

## Step 1: Create Render Blueprint

1. In Render, click `New +` -> `Blueprint`.
2. Connect your repo and choose the branch.
3. Confirm Render detects `render.yaml`.
4. Verify it creates:
- `adv-frontend` (static web)
- `adv-backend` (node web)
- `adv-python-doc-service` (docker web)
- `adv-mlflow` (private service)
- `adv-airflow-api` (private service)
- `adv-airflow-scheduler` (worker)
- `adv-airflow-dag-processor` (worker)
- `adv-airflow-db` (postgres)
- `adv-mlflow-db` (postgres)

## Step 2: Add Environment Variables in Render (Before First Deploy)

Add these in each service dashboard.

1. `adv-backend`
- `MONGO_URI`
- `JWT_SECRET`
- `DOC_SERVICE_BASE_URL` (set after Python service is deployed; temporary placeholder is fine)

2. `adv-python-doc-service`
- `GEMINI_API_KEY`
- `ES_URLS`
- `ES_API_KEY`
- `AWS_DEFAULT_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- Optional override: `MLFLOW_TRACKING_URI`, `AIRFLOW_API_BASE_URL`
- Optional auth: `AIRFLOW_API_TOKEN` or (`AIRFLOW_API_USERNAME` + `AIRFLOW_API_PASSWORD`)

3. `adv-mlflow`
- `MLFLOW_S3_BUCKET` (example: `mlflow-custom-care`)
- `AWS_DEFAULT_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

4. `adv-airflow-api`
- `AIRFLOW__CORE__FERNET_KEY` (generate once, long random base64/string)
- `AIRFLOW_ADMIN_USERNAME`
- `AIRFLOW_ADMIN_PASSWORD`
- `AIRFLOW_ADMIN_EMAIL`
- Optional defaults for DAG runs:
  - `AIRFLOW_ML_CLIENT_ID`
  - `AIRFLOW_ML_BOT_ID`

5. `adv-airflow-scheduler` and `adv-airflow-dag-processor`
- `AIRFLOW__CORE__FERNET_KEY` (must match `adv-airflow-api`)
- Optional defaults:
  - `AIRFLOW_ML_CLIENT_ID`
  - `AIRFLOW_ML_BOT_ID`

6. `adv-frontend`
- `REACT_APP_API_BASE_URL` (set after backend is live, e.g. `https://<backend>.onrender.com/api`)

## Step 3: Deploy Core Data Layer First

1. Trigger deploy for `adv-mlflow-db` and `adv-airflow-db` (Render usually provisions automatically from Blueprint).
2. Wait until both databases show `Available`.

## Step 4: Deploy MLflow

1. Deploy `adv-mlflow`.
2. Confirm logs show MLflow server started.
3. Health check from Render shell:
```bash
curl -sS http://localhost:5000/health
```
Expected: HTTP 200.

## Step 5: Deploy Python Doc Service

1. Deploy `adv-python-doc-service`.
2. Confirm `/docs` loads on its public URL.
3. This service auto-resolves internal MLflow/Airflow host/port from Render private services, so explicit URI vars are optional.

## Step 6: Deploy Backend

1. Set `DOC_SERVICE_BASE_URL` in `adv-backend` to Python service URL:
- `https://<adv-python-doc-service>.onrender.com`
2. Deploy `adv-backend`.
3. Test one API route from browser/Postman.

## Step 7: Deploy Frontend

1. Set `REACT_APP_API_BASE_URL`:
- `https://<adv-backend>.onrender.com/api`
2. Deploy `adv-frontend`.
3. Open app and test login/chat flow.

## Step 8: Deploy Airflow Components

Deploy in this order:
1. `adv-airflow-api`
- Runs DB migration and admin user creation during pre-deploy.
2. `adv-airflow-scheduler`
3. `adv-airflow-dag-processor`

Validation:
1. From `adv-airflow-api` shell:
```bash
curl -sS http://localhost:8080/api/v2/version
```
Expected: JSON with version info.

2. Trigger a DAG run via Airflow API (internal test):
```bash
curl -sS -X POST "http://localhost:8080/api/v2/dags/hdbscan_pipeline/dagRuns" \
  -H "Content-Type: application/json" \
  -u "${AIRFLOW_ADMIN_USERNAME}:${AIRFLOW_ADMIN_PASSWORD}" \
  -d "{\"conf\":{\"client_id\":\"<client_id>\",\"bot_id\":\"<bot_id>\"}}"
```

## Step 9: Enable Production Auto-Trigger

1. In `adv-python-doc-service`, set:
- `AIRFLOW_AUTOTRIGGER_ENABLED=true`
2. Redeploy `adv-python-doc-service`.

## Step 10: DVC One-Time Production Check

Run in `adv-python-doc-service` shell:
```bash
python -m dvc remote list
python -m dvc push
```

If remote is missing:
```bash
python -m dvc remote add -d prod s3://dvc-customcare/adv_project/dvc
python -m dvc remote modify prod region us-west-2
python -m dvc push
```

## 4. Secrets Policy (Important)

1. Never put real secrets in:
- `render.yaml`
- `.env.example`
- committed code
2. Enter secrets only in Render environment variables.
3. If a secret was ever exposed previously, rotate it before go-live.

## 5. Official References

- Render Blueprint spec: https://render.com/docs/blueprint-spec
- Render env vars (`sync: false`, `fromService`, `fromDatabase`): https://render.com/docs/blueprint-spec#environment-groups-and-environment-variables
- Airflow API security and auth: https://airflow.apache.org/docs/apache-airflow/stable/security/api.html
- Airflow CLI reference: https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html
- MLflow tracking server: https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/
- DVC S3 remote: https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3
