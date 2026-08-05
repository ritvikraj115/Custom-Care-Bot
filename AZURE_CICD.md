# Azure CI/CD Setup

This repo uses GitHub Actions for four jobs:

- `CI`: validates frontend, backend, and Python syntax/builds.
- `Deploy Frontend to Azure Static Web Apps`: deploys `chatbot`.
- `Deploy Backend to Azure App Service`: deploys `server`.
- `Deploy Python Doc Service to Azure Container Apps`: builds and deploys `python_doc_service`.

## GitHub Secrets

Go to:

```text
GitHub repo -> Settings -> Secrets and variables -> Actions -> Secrets
```

Add:

```text
AZURE_STATIC_WEB_APPS_API_TOKEN
AZURE_BACKEND_PUBLISH_PROFILE
AZURE_CREDENTIALS
```

Where they come from:

- `AZURE_STATIC_WEB_APPS_API_TOKEN`: Azure Static Web App -> Manage deployment token.
- `AZURE_BACKEND_PUBLISH_PROFILE`: Azure App Service backend -> Overview -> Download publish profile. Paste the full XML.
- `AZURE_CREDENTIALS`: service principal JSON for Azure CLI deploys.

Create service principal:

```powershell
$SUBSCRIPTION_ID=$(az account show --query id -o tsv)

az ad sp create-for-rbac `
  --name "sp-custom-care-bot-github" `
  --role contributor `
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-custom-care-bot" `
  --sdk-auth
```

Paste the full JSON output into `AZURE_CREDENTIALS`.

## GitHub Variables

Go to:

```text
GitHub repo -> Settings -> Secrets and variables -> Actions -> Variables
```

Add:

```text
REACT_APP_API_BASE_URL=https://<backend-app-name>.azurewebsites.net/api
AZURE_BACKEND_APP_NAME=<backend-app-name>
AZURE_RESOURCE_GROUP=rg-custom-care-bot
AZURE_CONTAINER_REGISTRY=<acr-name-without-.azurecr.io>
AZURE_DOC_CONTAINER_APP_NAME=custom-care-doc-service
```

## Azure App Settings

Backend App Service settings:

```text
NODE_ENV=production
MONGO_URI=<external MongoDB URI>
JWT_SECRET=<long random secret>
DOC_SERVICE_BASE_URL=https://<doc-container-app-fqdn>
CORS_ORIGIN=https://<static-web-app-name>.azurestaticapps.net
RAG_TIMEOUT_MS=120000
DEBUG_RAG_PAYLOAD=0
SOCIAL_REFRESH_INTERVAL_MS=21600000
SOCIAL_REFRESH_BATCH_LIMIT=50
```

Python Container App env vars:

```text
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
ES_ENABLED=false
AIRFLOW_AUTOTRIGGER_ENABLED=false
MLFLOW_ENABLED=false
DVC_AUTO_FLOW_ENABLED=false
RUN_QUALITY_CHECKS=false
WEBSITE_USE_PLAYWRIGHT=true
SOCIAL_USE_PLAYWRIGHT=true
```

## Manual Run

GitHub:

```text
Repo -> Actions -> select workflow -> Run workflow
```

## Health Checks

Backend:

```powershell
curl https://<backend-app-name>.azurewebsites.net/api/health
```

Python doc service:

```powershell
curl https://<doc-container-app-fqdn>/health
```

