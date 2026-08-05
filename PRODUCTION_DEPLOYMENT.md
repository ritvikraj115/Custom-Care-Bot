# Custom Care Bot Azure Production Deployment

This repo is now documented for one primary deployment path: Azure managed services.

We are not deploying every service together. The first production-style deployment uses only the services needed to run the product:

- React frontend: Azure Static Web Apps
- Node/Express API: Azure App Service
- Python document intelligence service: Azure Container Apps
- MongoDB: external MongoDB Atlas or your existing external MongoDB
- Optional search: existing external Elasticsearch if you already have it

Airflow, MLflow, DVC remotes, and full MLOps orchestration are valuable learning pieces, but they are not part of the first Azure deployment. Add them later after the core app is stable.

## 1. Why These Azure Services

Azure Static Web Apps is best for the React `chatbot` folder because it builds static files and gives HTTPS, routing, and GitHub deployment.

Azure App Service is best for the Node backend because it is a standard managed web app runtime. You do not manage Nginx, PM2, SSH, or VM patching.

Azure Container Apps is best for the Python doc service because this service is ML-heavy. It needs FAISS, PyTorch, TensorFlow, Playwright, sentence-transformers, spaCy, and native Linux dependencies. The existing Dockerfile already captures those dependencies, so a container is safer than trying to install everything directly on App Service.

MongoDB stays external because the backend uses Mongoose. Moving to Azure SQL would require a database rewrite. Moving to Cosmos DB for MongoDB can be a future learning task, not the first deployment.

Azure AI Foundry is the best next AI learning service. Use it later to replace Gemini or compare Gemini vs Azure-hosted models.

## 2. Current Repo Services

Frontend:

```text
chatbot/
```

Node backend:

```text
server/
```

Python AI/RAG/doc service:

```text
python_doc_service/
```

Optional MLOps stack:

```text
infra/airflow_mlflow_stack/
```

The optional MLOps stack remains in the repo for learning, but do not deploy it in the first Azure pass.

## 3. Target URLs

After deployment you will have:

```text
Frontend:
https://<static-web-app-name>.azurestaticapps.net

Backend:
https://<backend-app-name>.azurewebsites.net

Python doc service:
https://<doc-container-app-name>.<region>.azurecontainerapps.io
```

Frontend calls:

```text
https://<backend-app-name>.azurewebsites.net/api
```

Backend calls:

```text
https://<doc-container-app-url>
```

## 4. Azure CLI Setup

Install Azure CLI, then:

```powershell
az login
az account show
```

Set variables:

```powershell
$RG="rg-custom-care-bot"
$LOC="centralindia"
$PLAN="plan-custom-care-bot"
$BACKEND_APP="custom-care-api-<unique>"
$ACR="customcareacr<unique>"
$ENV="env-custom-care-bot"
$DOC_APP="custom-care-doc-service"
$SWA="custom-care-web-<unique>"
```

Use lowercase unique names. Replace `<unique>` with your name or random digits.

If `centralindia` has capacity issues, use:

```powershell
$LOC="eastus"
```

## 5. Create Resource Group

```powershell
az group create --name $RG --location $LOC
```

Theory: a resource group is the lifecycle boundary. If you want to delete everything later, delete the resource group.

## 6. Create Node Backend App Service

Create Linux App Service plan:

```powershell
az appservice plan create `
  --name $PLAN `
  --resource-group $RG `
  --location $LOC `
  --is-linux `
  --sku B1
```

Create backend web app:

```powershell
az webapp create `
  --name $BACKEND_APP `
  --resource-group $RG `
  --plan $PLAN `
  --runtime "NODE:22-lts"
```

If Azure says Node 22 is unavailable in your region, use:

```powershell
--runtime "NODE:20-lts"
```

Set startup command:

```powershell
az webapp config set `
  --name $BACKEND_APP `
  --resource-group $RG `
  --startup-file "node server.js"
```

Set backend environment variables:

```powershell
az webapp config appsettings set `
  --name $BACKEND_APP `
  --resource-group $RG `
  --settings `
    NODE_ENV=production `
    MONGO_URI="mongodb+srv://USER:PASSWORD@CLUSTER/DB?retryWrites=true&w=majority" `
    JWT_SECRET="replace-with-long-random-secret" `
    DOC_SERVICE_BASE_URL="https://placeholder-doc-service-url" `
    CORS_ORIGIN="https://placeholder-static-web-app-url" `
    RAG_TIMEOUT_MS=120000 `
    DEBUG_RAG_PAYLOAD=0 `
    SOCIAL_REFRESH_INTERVAL_MS=21600000 `
    SOCIAL_REFRESH_BATCH_LIMIT=50
```

You will replace `DOC_SERVICE_BASE_URL` and `CORS_ORIGIN` after creating those services.

## 7. Create Container Registry

Azure Container Registry stores your Python service Docker image.

```powershell
az acr create `
  --name $ACR `
  --resource-group $RG `
  --sku Basic `
  --admin-enabled true
```

Theory: Docker images are immutable deployable packages. For the Python service, this is cleaner than asking Azure to reinstall complex ML dependencies on every deploy.

## 8. Create Container Apps Environment

Install extension:

```powershell
az extension add --name containerapp --upgrade
```

Create environment:

```powershell
az containerapp env create `
  --name $ENV `
  --resource-group $RG `
  --location $LOC
```

Theory: a Container Apps environment is a managed boundary for one or more containerized apps. It handles ingress, revisions, scaling, and logs.

## 9. Build And Push Python Doc Service Image

Login to ACR:

```powershell
az acr login --name $ACR
```

Build image locally and push:

```powershell
cd "C:\Users\ritvi\Downloads\custom care bot\Custom-Care-Bot"

docker build `
  -t "$ACR.azurecr.io/custom-care-doc-service:latest" `
  .\python_doc_service

docker push "$ACR.azurecr.io/custom-care-doc-service:latest"
```

If you do not have Docker locally, use Azure build:

```powershell
az acr build `
  --registry $ACR `
  --image custom-care-doc-service:latest `
  .\python_doc_service
```

## 10. Create Python Doc Service Container App

Get ACR credentials:

```powershell
$ACR_USER=$(az acr credential show --name $ACR --query username -o tsv)
$ACR_PASS=$(az acr credential show --name $ACR --query "passwords[0].value" -o tsv)
```

Create container app:

```powershell
az containerapp create `
  --name $DOC_APP `
  --resource-group $RG `
  --environment $ENV `
  --image "$ACR.azurecr.io/custom-care-doc-service:latest" `
  --registry-server "$ACR.azurecr.io" `
  --registry-username $ACR_USER `
  --registry-password $ACR_PASS `
  --target-port 8000 `
  --ingress external `
  --cpu 2.0 `
  --memory 4Gi `
  --min-replicas 1 `
  --max-replicas 1 `
  --env-vars `
    GEMINI_API_KEY="your-gemini-key" `
    GEMINI_MODEL="gemini-2.5-flash" `
    GEMINI_TEMPERATURE="0.2" `
    ES_ENABLED="false" `
    AIRFLOW_AUTOTRIGGER_ENABLED="false" `
    MLFLOW_ENABLED="false" `
    DVC_AUTO_FLOW_ENABLED="false" `
    RUN_QUALITY_CHECKS="false" `
    WEBSITE_USE_PLAYWRIGHT="true" `
    SOCIAL_USE_PLAYWRIGHT="true"
```

Why disable Airflow/MLflow/DVC first:

- It lets the product run with document ingestion, RAG, autocomplete, and answer generation.
- It avoids deploying Postgres, Airflow, MLflow, DVC remote, and artifact storage on day one.
- You can enable them later as a focused MLOps phase.

Get doc service URL:

```powershell
$DOC_URL=$(az containerapp show `
  --name $DOC_APP `
  --resource-group $RG `
  --query properties.configuration.ingress.fqdn `
  -o tsv)

$DOC_URL="https://$DOC_URL"
echo $DOC_URL
```

Test:

```powershell
curl "$DOC_URL/health"
curl "$DOC_URL/docs"
```

The first startup can be slow because the container loads embedding models.

## 11. Update Backend With Doc Service URL

```powershell
az webapp config appsettings set `
  --name $BACKEND_APP `
  --resource-group $RG `
  --settings DOC_SERVICE_BASE_URL="$DOC_URL"
```

Restart backend:

```powershell
az webapp restart --name $BACKEND_APP --resource-group $RG
```

Test backend:

```powershell
curl "https://$BACKEND_APP.azurewebsites.net/api/health"
```

## 12. Create Azure Static Web App

The cleanest way is Azure Portal:

1. Open Azure Portal.
2. Search `Static Web Apps`.
3. Create.
4. Resource group: `rg-custom-care-bot`.
5. Name: your `$SWA`.
6. Plan: Free is okay for learning.
7. Deployment source: GitHub.
8. Repository: `Custom-Care-Bot`.
9. Branch: `main`.
10. Build preset: React.
11. App location:

```text
chatbot
```

12. API location:

```text

```

Leave empty.

13. Output location:

```text
build
```

Azure will create a GitHub Actions workflow or give you a deployment token.

Set frontend environment variable:

```text
REACT_APP_API_BASE_URL=https://<backend-app-name>.azurewebsites.net/api
```

Important: this frontend env value must include `/api`.

## 13. Update Backend CORS With Frontend URL

After Static Web Apps is created, get the frontend URL:

```text
https://<static-web-app-name>.azurestaticapps.net
```

Then:

```powershell
az webapp config appsettings set `
  --name $BACKEND_APP `
  --resource-group $RG `
  --settings CORS_ORIGIN="https://<static-web-app-name>.azurestaticapps.net"

az webapp restart --name $BACKEND_APP --resource-group $RG
```

## 14. GitHub Actions CI/CD

This repo should use three deploy workflows:

- frontend deploy only when `chatbot/**` changes
- backend deploy only when `server/**` changes
- Python doc service deploy only when `python_doc_service/**` changes

Secrets and variables are described in `AZURE_CICD.md`.

## 15. Logs

Backend logs:

```powershell
az webapp log config `
  --name $BACKEND_APP `
  --resource-group $RG `
  --application-logging filesystem `
  --level information

az webapp log tail --name $BACKEND_APP --resource-group $RG
```

Python doc service logs:

```powershell
az containerapp logs show `
  --name $DOC_APP `
  --resource-group $RG `
  --follow
```

## 16. What To Test

1. Frontend opens.
2. Register works.
3. Login works.
4. Create bot works.
5. Upload document works.
6. Ask public chat question works.
7. Escalation/feedback flow works.

Health checks:

```powershell
curl "https://$BACKEND_APP.azurewebsites.net/api/health"
curl "$DOC_URL/health"
```

## 17. Optional Industry Learning After Core Deploy

Do these later, one at a time:

1. Azure AI Foundry
   - Replace Gemini calls in `python_doc_service/app/main.py` with Azure-hosted model inference.
   - Learn model deployment, keys, endpoints, and responsible AI controls.

2. Azure Application Insights
   - Add structured logs, traces, latency, and failures.
   - This is very industry-relevant DevOps/observability.

3. Azure Storage Account
   - Persist uploaded PDFs, generated indexes, and model artifacts.
   - This is better than container-local filesystem storage.

4. Azure Container Apps Jobs
   - Move ingestion/training jobs out of live request handling.
   - This is a strong production architecture for ML workloads.

5. Azure Machine Learning or MLflow on Azure
   - Track experiments and artifacts once the app is stable.

Do not add all of these together. Add one, test, understand, then continue.

## 18. Cost Control

Create a budget:

```text
Azure Portal -> Cost Management -> Budgets -> Add
```

Start low:

```text
$10-$25 monthly
```

Stop backend:

```powershell
az webapp stop --name $BACKEND_APP --resource-group $RG
```

Scale Python doc service to zero when not testing:

```powershell
az containerapp update `
  --name $DOC_APP `
  --resource-group $RG `
  --min-replicas 0
```

Delete all Azure resources:

```powershell
az group delete --name $RG
```

## 19. References

- Azure Static Web Apps React deployment: https://learn.microsoft.com/en-us/azure/static-web-apps/deploy-react
- Azure Static Web Apps build configuration: https://learn.microsoft.com/en-us/azure/static-web-apps/build-configuration
- Azure App Service GitHub Actions deployment: https://learn.microsoft.com/en-us/azure/app-service/deploy-github-actions
- Azure App Service custom container GitHub Actions: https://learn.microsoft.com/en-us/azure/app-service/deploy-container-github-action
- Microsoft Foundry Models deployment overview: https://learn.microsoft.com/en-us/azure/foundry/concepts/deployments-overview

