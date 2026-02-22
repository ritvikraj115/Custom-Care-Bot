# Python Doc Service

## DVC Pipeline (Clustering + Autocomplete)

This repo now includes a bot-specific DVC pipeline with 2 stages:

1. `cluster_train`: runs document clustering/index build.
2. `autocomplete_track`: tracks the latest trained autocomplete artifacts.

### Files added for DVC

- `dvc.yaml`
- `params.yaml`
- `data/dvc/pdf_manifest.json`
- `scripts/dvc/run_cluster_train.py`
- `scripts/dvc/run_autocomplete_train.py`
- `scripts/dvc/track_autocomplete_artifacts.py`
- `requirements-dvc.txt`

### Setup

1. Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

2. Install DVC:

```powershell
pip install -r requirements-dvc.txt
```

3. Initialize DVC in this folder:

If this repo uses Git:

```powershell
python -m dvc init
```

If this folder is not a Git repo:

```powershell
python -m dvc init --no-scm
```

4. Edit `params.yaml`:
- Set `dvc.client_id`
- Set `dvc.bot_id`
- Keep or change `dvc.rebuild_mode`

5. Edit `data/dvc/pdf_manifest.json`:
- Add your PDF entries under `pdfs`.
- Each entry can be a string path.
- Or each entry can be an object like `{"path":"../../server/uploads/your.pdf","source_type":"upload"}`.
- For website snapshots use `{"path":"../../server/uploads/site.pdf","source_type":"website","source_url":"https://example.com"}`.

### Run

```powershell
python -m dvc repro
```

This produces:

- `storage/indexes/client_<client_id>/bot_<bot_id>`
- `storage/autocomplete/client_<client_id>/bot_<bot_id>`
- `storage/dvc_runs/client_<client_id>/bot_<bot_id>/clustering_summary.json`

### Automatic DVC on retrain trigger

When autocomplete retraining is triggered by existing logic (bootstrap or pending-question thresholds), the service now runs:

```powershell
python -m dvc repro autocomplete_track
```

The service updates `params.yaml` with the current `client_id` and `bot_id` before running this command, so no manual edit is required.

Environment flags:

- `DVC_AUTO_FLOW_ENABLED=true` (default): enable/disable auto DVC tracking.
- `DVC_AUTO_TIMEOUT_SEC=900` (default): timeout for each DVC command.
- `DVC_AUTO_PUSH=false` (default): if `true`, run `python -m dvc push` after tracking.

## GitHub Actions Automation

- `HDBSCAN Model Flow`: `.github/workflows/hdbscan-model-flow.yml`
- `Autocomplete Model Flow`: `.github/workflows/autocomplete-model-flow.yml`

Both workflows support `workflow_dispatch` with bot/client inputs and can also run on code/data path updates.

## Airflow Orchestration

- `hdbscan_pipeline`: `airflow/dags/hdbscan_pipeline_dag.py`
  - Ingest + cluster + deploy.
- `autocomplete_pipeline`: `airflow/dags/autocomplete_pipeline_dag.py`
  - DVC pull + retrain + track + deploy.

Required env vars for DAG runs:

- `AIRFLOW_ML_CLIENT_ID`
- `AIRFLOW_ML_BOT_ID`
- Optional: `AIRFLOW_HDBSCAN_REBUILD_MODE`, `AIRFLOW_HDBSCAN_PDF_MANIFEST`, `AIRFLOW_DEPLOY_CMD`

## Monitoring Dashboard

Drift/monitoring stats are logged to `storage/monitoring/model_dashboard.json` and exposed via:

- `GET /monitoring/model-dashboard`
