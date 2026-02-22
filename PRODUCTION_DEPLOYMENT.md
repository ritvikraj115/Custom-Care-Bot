# Render Deployment Modes

This repo now has two Render blueprints:

1. `render.yaml` -> no-payment mode (free-tier friendly):
- `adv-frontend` (static)
- `adv-backend` (free web)
- `adv-python-doc-service` (free web)

2. `render.paid.yaml` -> full stack mode (requires payment method):
- Adds MLflow, Airflow API/scheduler/dag-processor, and Postgres DBs.

## Why Render asked for payment

Your previous blueprint included paid-only components:
- `type: pserv` private services
- `type: worker`
- managed Postgres databases
- persistent disk

So Render blocked deployment until billing info was added.

## No-Pay Deploy Steps

1. In Render, create Blueprint using `render.yaml`.
2. Add env vars in dashboard (do not commit secrets):
- `adv-backend`: `MONGO_URI`, `JWT_SECRET`, `DOC_SERVICE_BASE_URL`
- `adv-python-doc-service`: `GEMINI_API_KEY`, `ES_URLS`, `ES_API_KEY`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `adv-frontend`: `REACT_APP_API_BASE_URL`
3. Deploy `adv-python-doc-service`.
4. Set backend `DOC_SERVICE_BASE_URL` to Python service URL and deploy backend.
5. Set frontend `REACT_APP_API_BASE_URL` to backend `/api` URL and deploy frontend.

## No-Pay Limitations

1. Airflow and MLflow are disabled in `render.yaml`.
2. No persistent disk for Python service.
3. Free web services can sleep and cold-start.

If you later decide to enable full MLOps, switch to `render.paid.yaml`.
