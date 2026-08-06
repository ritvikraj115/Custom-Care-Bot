# Custom Care Bot

Custom Care Bot is a multi-tenant customer-support chatbot platform. It lets a business create bots, upload PDFs, connect website/social context, answer public user questions, collect feedback, escalate weak answers, and improve retrieval/autocomplete behavior over time.

The project is intentionally more than a simple "chat with PDF" app. It combines product workflows, RAG, graph-based answer routing, semantic memory, hybrid retrieval, bot-specific autocomplete, and MLOps-style training/artifact tracking.

## 1. Product Idea

Most support chatbots fail in predictable ways:

- They answer from stale or incomplete documents.
- They hallucinate when retrieval confidence is low.
- They do not learn from repeated bad answers.
- They treat every customer/business the same.
- They do not expose enough operational signals to improve the system.

This project addresses those problems with a platform-style design:

- each client can create multiple bots
- each bot owns its documents, website/social sources, chat history, feedback, and analytics
- answers are grounded in uploaded/business content
- low-confidence or repeatedly disliked answers can escalate to humans
- experience memory helps the system reuse or avoid past answer patterns
- autocomplete learns bot-specific customer question patterns
- MLOps tooling tracks ingestion, clustering, autocomplete artifacts, and model quality signals

## 2. High-Level Architecture

```text
Browser
  -> React frontend
      -> Node/Express backend
          -> MongoDB
          -> Python document intelligence service
              -> PDF/website ingestion
              -> embedding and chunking
              -> FAISS vector indexes
              -> optional Elasticsearch hybrid retrieval
              -> LangGraph answer orchestration
              -> Gemini LLM
              -> DVC / MLflow / Airflow hooks
```

Main folders:

```text
chatbot/                         React frontend
server/                          Node/Express backend
python_doc_service/              FastAPI RAG, retrieval, ML, MLOps service
infra/airflow_mlflow_stack/       All-in-one Docker Compose stack
```

## 3. Services

### Frontend: `chatbot/`

The frontend is a React app for:

- registration and login
- dashboard
- bot creation
- document upload
- bot analytics
- public chat
- feedback and escalation UX

Key frontend details:

- React 19
- React Router
- Axios API client
- Markdown answer rendering
- environment variable: `REACT_APP_API_BASE_URL`

The frontend talks only to the Node API. It should not directly call the Python service in production.

### Backend: `server/`

The backend owns product and tenant state.

Responsibilities:

- authentication and JWT validation
- client/tenant isolation
- bot CRUD
- document metadata
- upload queueing
- chat sessions and messages
- feedback tracking
- escalation tracking
- analytics
- orchestration calls to the Python service

Important backend routes:

```text
/api/auth
/api/bots
/api/documents
/api/chat
/api/escalations
/api/health
```

Important backend environment variables:

```text
PORT=5000
MONGO_URI=<external MongoDB URI>
JWT_SECRET=<strong secret>
DOC_SERVICE_BASE_URL=http://python-doc-service:8000
RAG_TIMEOUT_MS=120000
DEBUG_RAG_PAYLOAD=0
SOCIAL_REFRESH_INTERVAL_MS=21600000
SOCIAL_REFRESH_BATCH_LIMIT=50
```

### Python Doc Service: `python_doc_service/`

The Python service owns the intelligence layer.

Responsibilities:

- PDF ingestion
- website crawling and content conversion
- document extraction
- boilerplate-aware chunking
- embeddings
- UMAP + HDBSCAN clustering
- hierarchical FAISS indexing
- optional Elasticsearch hybrid retrieval
- LangGraph answer routing
- Gemini answer generation
- experience vector memory
- bot-specific autocomplete
- monitoring dashboard
- DVC, MLflow, and Airflow integration hooks

Important endpoints:

```text
GET  /health
GET  /docs
POST /process
POST /answer
POST /autocomplete/suggest
POST /autocomplete/train
POST /autocomplete/status
POST /autocomplete/top-questions
POST /experience/index
POST /experience/search
POST /experience/update-feedback
POST /bot/delete
GET  /monitoring/model-dashboard
POST /social/refresh
GET  /social/status/{bot_id}
```

Important Python environment variables:

```text
GEMINI_API_KEY=<Gemini key>
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2
ES_ENABLED=true
ES_URLS=<Elasticsearch URL>
ES_API_KEY=<Elasticsearch API key>
DVC_AUTO_FLOW_ENABLED=true
DVC_AUTO_PUSH=true
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=http://mlflow:5000
AIRFLOW_AUTOTRIGGER_ENABLED=true
AIRFLOW_API_BASE_URL=http://airflow-apiserver:8080
```

## 4. Request And Data Flow

### Login / Product Flow

```text
React
  -> Node /api/auth
  -> MongoDB Client model
  -> JWT returned
  -> React stores token
  -> future API calls include Authorization header
```

### Bot Creation

```text
React create bot form
  -> Node /api/bots
  -> MongoDB Bot model
  -> optional social refresh starts in background
```

### Document Upload And Ingestion

```text
React uploads PDFs
  -> Node /api/documents/upload
  -> Node stores Document rows in MongoDB
  -> Node deduplicates by file hash
  -> Node enqueues ingestion job
  -> Node calls Python /process
  -> Python builds runtime PDF manifest
  -> DVC pipeline may run
  -> extraction/chunking/embedding/clustering happens
  -> FAISS indexes and summaries are written
  -> MLflow logs metrics if enabled
  -> Airflow DAG trigger may run if enabled
  -> Python reloads indexes
  -> frontend polls job status
```

### User Chat

```text
Public user asks question
  -> React chat UI
  -> Node /api/chat/message
  -> Python /answer
  -> LangGraph chooses route
  -> retrieval searches docs/memory/social as needed
  -> Gemini generates answer from grounded context
  -> Python returns answer, chunks, confidence, references
  -> Node stores ChatMessage and Experience records
  -> React renders answer
```

### Feedback And Escalation

```text
User gives negative feedback
  -> Node stores feedback
  -> Python updates experience vector memory
  -> repeated/low-confidence failures become escalation candidates
  -> owner can resolve and improve future behavior
```

## 5. Technical Design

### Structure-Aware Document Ingestion

The Python service does not treat PDFs as plain text blobs.

It extracts blocks, detects headings, groups text into sections, and then chunks content. This preserves more document structure than fixed-size chunking.

Why it matters:

- support PDFs often contain headings, policy sections, tables, repeated disclaimers, and footer noise
- naive chunking can mix unrelated sections
- structured chunks improve retrieval and answer grounding

### Boilerplate-Aware Chunking

The pipeline filters repeated low-value text using:

- TF-IDF-style salience
- semantic centrality
- cross-section repetition
- local chunk-level cleanup

Tradeoff:

- stronger retrieval quality
- more pipeline complexity
- possible risk of removing text that looks repetitive but is legally important

### Embeddings And Vector Search

The service uses:

```text
SentenceTransformer("all-MiniLM-L6-v2")
FAISS
```

Why:

- lightweight enough for CPU serving
- good quality-to-latency baseline
- useful for chunk search, experience memory, and autocomplete dedupe

Tradeoff:

- not domain-specialized
- may miss rare technical vocabulary
- generated indexes must be persisted carefully in production

### UMAP + HDBSCAN Topic Clustering

The document pipeline uses UMAP and HDBSCAN to identify topic clusters.

Why:

- support content has uneven topic density
- HDBSCAN can mark noise instead of forcing every chunk into a cluster
- cluster routing can reduce retrieval search space

Tradeoff:

- more moving parts than plain vector search
- parameter sensitivity
- clustering quality must be monitored

### Hierarchical Retrieval

Retrieval is not only "search all chunks".

The design can route queries to clusters first, then search chunks within relevant clusters.

Why:

- faster search on larger bot corpora
- more interpretable topic-level behavior
- better analytics around weak/unclear topics

### Hybrid Retrieval With Elasticsearch

Elasticsearch is optional but valuable.

Semantic retrieval helps with paraphrases. BM25 helps with exact names, policy terms, URLs, IDs, and rare keywords.

Final retrieval can combine:

```text
FAISS semantic score
Elasticsearch BM25 score
tenant/bot filters
source metadata
```

Tradeoff:

- stronger production retrieval
- extra external dependency
- mapping/index management
- must preserve strict tenant isolation filters

### LangGraph Answer Orchestration

The answer path uses graph-style routing rather than one linear RAG chain.

The graph can decide between:

- normal document lookup
- memory reuse
- clarification
- social/website freshness
- secondary retrieval
- owner/human escalation

Why:

- support conversations need control flow
- low-confidence behavior should be explicit
- feedback-aware retries need different behavior from first answers

### Experience Memory

Experience memory stores semantically searchable past interactions.

It lets the system:

- reuse good previous answers
- avoid repeating bad answers
- notice repeated negative patterns
- escalate recurring unresolved questions

Tradeoff:

- improves adaptation without full retraining
- requires careful feedback handling
- can preserve stale behavior if not aged or moderated

### Bot-Specific Autocomplete

The autocomplete system is not just a static list.

It combines:

- bot-specific question logs
- seed questions generated from docs and website content
- semantic dedupe
- recency weighting
- a lightweight custom sequence model
- optional Elasticsearch lexical recall

Why:

- each business has different customer language
- autocomplete improves UX and captures demand signals
- frequent questions become visible product intelligence

## 6. MLOps Design

The MLOps layer exists because ingestion and training are not ordinary API calls.

### DVC

DVC versions data-like artifacts outside Git:

- generated indexes
- clustering summaries
- autocomplete artifacts
- bot/client-specific pipeline outputs

Current stages:

```text
cluster_train
autocomplete_track
```

### MLflow

MLflow tracks:

- pipeline parameters
- clustering metrics
- autocomplete metrics
- run duration
- artifacts
- client and bot tags

Why:

- compare runs
- debug regressions
- understand whether retrieval/autocomplete improved

### Airflow

Airflow is used for orchestration:

- trigger HDBSCAN pipeline
- trigger autocomplete pipeline
- provide replayable DAG runs
- separate training control from web request code

### Postgres

The all-in-one stack uses Postgres for:

- Airflow metadata
- MLflow metadata

MongoDB remains the product database.

## 7. What Makes This Project Unique

This project is unique because it combines several production ideas in one support-bot platform:

1. Multi-tenant bot creation rather than one hardcoded assistant.
2. PDF plus website/social context.
3. Structure-aware ingestion instead of naive text splitting.
4. UMAP/HDBSCAN clustering before retrieval.
5. Hierarchical FAISS search.
6. Optional Elasticsearch hybrid retrieval.
7. LangGraph answer routing.
8. Experience memory from real feedback.
9. Human escalation as a first-class safety mechanism.
10. Bot-specific autocomplete learning.
11. DVC, MLflow, Airflow, and monitoring hooks.

The core idea is that a useful enterprise support bot needs retrieval, memory, feedback, freshness, safety, and operations, not just an LLM prompt.

## 8. Design Tradeoffs

### Single VM Docker Stack

Pros:

- easiest to understand end to end
- all services communicate on a private Docker network
- simple deployment with one `docker compose up -d --build`
- good for demos, learning, and interviews

Cons:

- one machine can become a bottleneck
- Airflow, MLflow, training, and serving compete for CPU/RAM
- local Docker volumes are not enough for serious artifact durability
- horizontal scaling is hard because some indexes are process-local

### Managed Azure Split

Pros:

- better long-term operational shape
- frontend, backend, and ML service deploy independently
- Container Apps can later scale jobs separately
- easier to add Application Insights, Blob Storage, and queues

Cons:

- more cloud services to configure
- more networking and secret-management concepts
- requires deeper DevOps discipline

### Hybrid Search

Pros:

- improves exact keyword and semantic recall
- better for real support docs with names, SKUs, URLs, and policies

Cons:

- extra service dependency
- index mapping/versioning risk
- more ranking tuning

### Runtime Training

Pros:

- simple user experience after document upload
- immediate index refresh

Cons:

- can cause request timeouts
- can slow live chat
- should eventually move to background jobs

## 9. Recommended Deployment Options

There are two supported deployment mindsets:

1. All-in-one EC2/Azure VM Docker deployment
   - best for learning the entire system
   - runs frontend, backend, Python, Airflow, MLflow, and Postgres together

2. Azure managed deployment
   - best for cleaner cloud production learning
   - documented in [PRODUCTION_DEPLOYMENT.md](./PRODUCTION_DEPLOYMENT.md)

MLOps expansion is documented in [MLOPS_DEPLOYMENT.md](./MLOPS_DEPLOYMENT.md).

## 10. All-In-One EC2 Deployment

This is the earlier deployment style: run every service on one EC2 Ubuntu VM using Docker Compose.

It is useful when you want to understand how all services communicate without learning many managed cloud products at once.

### 10.1 EC2 Instance

Recommended EC2:

```text
AMI: Ubuntu Server 22.04 LTS or 24.04 LTS
Instance type: at least 4 vCPU / 16 GB RAM
Disk: 80 GB gp3 minimum
```

Why that size:

- Python service loads embedding models, FAISS, TensorFlow/PyTorch-related packages, Playwright, and spaCy
- Airflow has multiple processes
- MLflow and Postgres also run
- builds can be memory-heavy

For a small demo you can try smaller, but expect build/runtime failures if memory is too low.

### 10.2 Security Group

Open only what you need.

For testing:

```text
22    SSH             your IP only
80    frontend        0.0.0.0/0
5000  backend API     your IP only, or closed if frontend proxies API
8000  Python docs     your IP only
8091  Airflow         your IP only
5001  MLflow          your IP only
```

Do not leave Airflow or MLflow open publicly in a real deployment.

Better production shape:

```text
22 only your IP
80/443 public
Airflow/MLflow behind VPN, private network, or IP allowlist
```

### 10.3 Install Docker On EC2

SSH into the VM:

```bash
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
```

Install Docker:

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

Verify:

```bash
docker --version
docker compose version
```

### 10.4 Clone The Repo

```bash
git clone <your-github-repo-url> Custom-Care-Bot
cd Custom-Care-Bot
```

### 10.5 Configure Environment

Go to the Docker stack folder:

```bash
cd infra/airflow_mlflow_stack
cp .env.example .env
nano .env
```

Fill these required values:

```text
MONGO_URI=<external MongoDB URI>
JWT_SECRET=<strong random secret>
GEMINI_API_KEY=<Gemini API key>
ES_URLS=<Elasticsearch URL, if using hybrid search>
ES_API_KEY=<Elasticsearch API key, if using hybrid search>
AWS_ACCESS_KEY_ID=<AWS key for S3 artifacts if using DVC/MLflow S3>
AWS_SECRET_ACCESS_KEY=<AWS secret for S3 artifacts if using DVC/MLflow S3>
AIRFLOW_FERNET_KEY=<generated fernet key>
AIRFLOW_API_AUTH_JWT_SECRET=<strong random secret>
AIRFLOW_API_SECRET_KEY=<strong random secret>
AIRFLOW_ADMIN_PASSWORD=<change default>
```

Generate an Airflow fernet key:

```bash
docker run --rm apache/airflow:3.1.7 python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

Generate random secrets:

```bash
openssl rand -hex 32
```

If you do not want Elasticsearch initially:

```text
ES_ENABLED=false
ES_URLS=
ES_API_KEY=
```

If you do not want DVC/MLflow artifact push initially:

```text
DVC_AUTO_PUSH=false
MLFLOW_ENABLED=false
```

### 10.6 What The Compose Stack Starts

The file:

```text
infra/airflow_mlflow_stack/docker-compose.yml
```

starts:

```text
frontend               React build served by Nginx
backend                Node/Express API
python-doc-service     FastAPI RAG/ML service
postgres               metadata DB for Airflow and MLflow
mlflow                 experiment tracking server
airflow-init           one-time DB/user setup
airflow-apiserver      Airflow API/UI
airflow-scheduler      DAG scheduler
airflow-dag-processor  DAG parser/processor
```

Internal service communication:

```text
frontend -> backend: http://backend:5000
backend -> python-doc-service: http://python-doc-service:8000
python-doc-service -> airflow-apiserver: http://airflow-apiserver:8080
python-doc-service -> mlflow: http://mlflow:5000
mlflow -> postgres: postgres:5432
airflow -> postgres: postgres:5432
```

External ports:

```text
80    frontend
5000  backend
8000  Python FastAPI docs/service
8091  Airflow
5001  MLflow
5433  Postgres host mapping
```

### 10.7 Start Everything

From:

```bash
cd infra/airflow_mlflow_stack
```

Run:

```bash
docker compose up -d --build
```

First build can take time because the Python image installs ML packages, Playwright Chromium, spaCy model, and sentence-transformer model.

Check services:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f --tail=150
```

Logs for one service:

```bash
docker compose logs -f backend
docker compose logs -f python-doc-service
docker compose logs -f airflow-apiserver
docker compose logs -f mlflow
```

### 10.8 Verify Endpoints

Open:

```text
Frontend:
http://EC2_PUBLIC_IP/

Backend:
http://EC2_PUBLIC_IP:5000/

Python docs:
http://EC2_PUBLIC_IP:8000/docs

Airflow:
http://EC2_PUBLIC_IP:8091

MLflow:
http://EC2_PUBLIC_IP:5001
```

Run smoke test:

```bash
chmod +x smoke-test.sh
./smoke-test.sh
```

The smoke test checks:

- frontend reachable
- backend reachable
- Python docs healthy
- MLflow health
- Airflow API
- backend-to-Python Docker-network connectivity

### 10.9 Airflow Login

Use values from `.env`:

```text
URL: http://EC2_PUBLIC_IP:8091
Username: AIRFLOW_ADMIN_USERNAME
Password: AIRFLOW_ADMIN_PASSWORD
```

Default example:

```text
airflow / change-me
```

Change the password before exposing this beyond your IP.

### 10.10 MLflow

Open:

```text
http://EC2_PUBLIC_IP:5001
```

Expected experiments:

```text
bot-pipeline
bot-autocomplete
```

When enabled, Python logs runs through:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### 10.11 DVC Setup

The Python service includes DVC files:

```text
python_doc_service/dvc.yaml
python_doc_service/params.yaml
```

The Compose stack expects S3-style artifact storage when DVC push is enabled.

Inside the stack:

```bash
docker compose exec python-doc-service bash
```

Then configure DVC remote if needed:

```bash
python -m dvc remote remove localstore || true
python -m dvc remote add -d prod s3://dvc-customcare/adv_project/dvc || true
python -m dvc remote modify prod region us-west-2
python -m dvc push
```

Use your own bucket and region. The bucket must exist and credentials must be valid in `.env`.

### 10.12 Updating The EC2 Deployment

From repo root:

```bash
git pull
cd infra/airflow_mlflow_stack
docker compose up -d --build
```

Restart one service:

```bash
docker compose restart backend
docker compose restart python-doc-service
docker compose restart airflow-apiserver
```

Stop all:

```bash
docker compose down
```

Stop and delete volumes:

```bash
docker compose down -v
```

Warning: `down -v` deletes Postgres and local volume data.

### 10.13 Common EC2 Problems

Python build fails:

- VM too small
- not enough disk
- Docker build timed out
- Playwright dependency install failed

Fix:

```text
Use 4 vCPU / 16 GB RAM
Use 80+ GB disk
Check docker compose logs python-doc-service
```

Mongo fails:

- bad `MONGO_URI`
- MongoDB Atlas IP allowlist does not include EC2 public IP
- database user/password wrong

Airflow unhealthy:

- missing `AIRFLOW_FERNET_KEY`
- bad Airflow secrets
- Postgres still starting

MLflow unhealthy:

- Postgres not ready
- bad artifact storage credentials
- S3 bucket missing

Frontend cannot call backend:

- wrong frontend API base URL
- backend port closed
- CORS/proxy issue

## 11. Local Development

Typical local development is three processes:

Backend:

```bash
cd server
npm install
npm run dev
```

Python service:

```bash
cd python_doc_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd chatbot
npm install
npm start
```

Local env examples:

```text
server/.env.example
python_doc_service/.env.example
chatbot/.env.example
```

## 12. Related Docs

- [PRODUCTION_DEPLOYMENT.md](./PRODUCTION_DEPLOYMENT.md): Azure managed deployment
- [AZURE_CICD.md](./AZURE_CICD.md): GitHub Actions + Azure setup
- [MLOPS_DEPLOYMENT.md](./MLOPS_DEPLOYMENT.md): staged MLOps deployment plan
- [PYTHON_DOC_SERVICE_MLOPS_INTERVIEW_GUIDE.md](./PYTHON_DOC_SERVICE_MLOPS_INTERVIEW_GUIDE.md): deep technical/interview explanation
- [infra/airflow_mlflow_stack/README.md](./infra/airflow_mlflow_stack/README.md): all-in-one Docker stack quick reference

## 13. One-Line Summary

Custom Care Bot is a production-style customer support AI platform that combines multi-tenant product workflows, retrieval-augmented generation, feedback-aware memory, human escalation, bot-specific autocomplete, and MLOps discipline around document intelligence.

