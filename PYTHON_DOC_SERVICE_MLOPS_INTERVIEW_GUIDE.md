# Python Doc Service, Elasticsearch, and MLOps Interview Guide

## 1. What This Project Is

This project is a multi-service customer support chatbot platform with:

- a React frontend for bot creation, analytics, and public chat
- a Node/Express backend for tenancy, auth, sessions, feedback, escalations, and orchestration
- a Python document intelligence service for ingestion, retrieval, answer generation, autocomplete, clustering, monitoring, and ML operations
- an MLOps stack using DVC, MLflow, Airflow, Postgres, Docker Compose, and S3-backed artifact storage

The most important technical idea is that this is not just "upload PDFs and ask questions".
It combines:

- structure-aware ingestion
- semantic plus lexical retrieval
- graph-based answer routing
- experience memory from past conversations
- human escalation when confidence/feedback is poor
- a separate autocomplete learning system
- operational MLOps for reproducibility and artifact tracking

## 2. Best 60-Second Interview Pitch

"I built a multi-tenant support chatbot platform where each bot can ingest its own PDFs and website content, build a clustered semantic index, answer questions with a graph-based RAG pipeline, learn from user interactions through experience memory, and improve the search UX with a bot-specific autocomplete model. I treated the Python doc service like a real ML system, so I added DVC for artifact versioning, MLflow for run tracking, Airflow for orchestration, monitoring for drift-like behavior, and Elasticsearch for hybrid retrieval and autocomplete suggestion recall. The core design goal was to make the assistant safer and more practical than plain RAG by combining retrieval, feedback-aware memory, social/web freshness, and human fallback."

## 3. Production Setup

### 3.1 Deployment shape

Production is designed as a single Linux VM Docker Compose stack.

Main services:

- `frontend` on port 80
- `backend` on port 5000
- `python-doc-service` on port 8000
- `airflow-apiserver`, `airflow-scheduler`, `airflow-dag-processor`
- `mlflow` on port 5001
- `postgres` shared by Airflow and MLflow

Code reference:

- `infra/airflow_mlflow_stack/docker-compose.yml`
- `infra/airflow_mlflow_stack/.env.example`
- `PRODUCTION_DEPLOYMENT.md`

### 3.2 Important production infrastructure decisions

1. Elasticsearch is not self-hosted in Docker here.
   It is expected as an external managed service via `ES_URLS` and `ES_API_KEY`.
   This is a strong production choice because search should be more durable and separately scalable than the VM app containers.

2. MLflow is self-hosted in the stack.
   The tracking server runs in Docker, stores metadata in Postgres, and pushes artifacts to S3.

3. Airflow is self-hosted in the stack.
   It is used as the orchestration layer for retraining/redeployment, but the Python service can still run local training directly as a fallback.

4. DVC runs inside the Python service environment.
   DVC stages are triggered from application code and from Airflow-style automation flows.

5. Persistent model/index state is stored in Docker volume-backed `storage/`.
   That includes hierarchical indexes, autocomplete models, social cache, and monitoring dashboard JSON.

### 3.3 Production risks in this setup

- Single VM means resource contention between API, training, Airflow, and MLflow.
- TensorFlow, FAISS, Playwright, and embedding models make the Python container heavy.
- Airflow, MLflow, and app workloads share one Postgres instance, which is simpler but creates blast-radius coupling.
- If the VM dies, local volume-backed artifacts disappear unless DVC/S3 pushes are reliable.
- Horizontal scaling is hard because some indexes are loaded in-memory per container process.

## 4. Python Doc Service Deep Dive

Primary entry point:

- `python_doc_service/app/main.py`

This service has two major roles:

1. online inference APIs
2. offline or semi-offline ingestion/training/orchestration

### 4.1 Ingestion flow

Main endpoint:

- `POST /process`

End-to-end flow:

1. Accept uploaded PDFs and optional website URL.
2. Save files to temp storage.
3. Optionally crawl website and convert it into PDF-like content.
4. Build a runtime PDF manifest for DVC.
5. Try DVC-driven clustering pipeline first.
6. Reload indexes after success.
7. Fall back to direct `run_pipeline()` if DVC path fails.
8. Trigger Airflow DAG for HDBSCAN pipeline asynchronously.
9. Bootstrap autocomplete seed questions from document and website content.
10. Return ingestion summary.

Why this is good:

- production path prefers reproducible pipeline execution
- app still works if orchestration layer is down
- website content and PDF content are unified into one training/indexing flow

### 4.2 Document extraction design

Main files:

- `app/pipeline/extract.py`
- `app/pipeline/chunk.py`

Extraction logic:

- Uses PyMuPDF (`fitz`) to read PDF blocks.
- Tracks average font size to detect headings.
- Builds sections from heading/content structure.
- Uses spaCy sentence segmentation.

My interview explanation:

"I did not treat PDFs as plain text blobs. I first extracted layout-aware blocks, then inferred headings from typography and short uppercase patterns, then grouped content into sections. That preserves document structure and gives better downstream chunks than naive fixed-window splitting."

### 4.3 Boilerplate-aware chunking

This is one of the most important uniqueness points.

The pipeline removes repeated low-value text in two layers:

1. global boilerplate detection across sections
2. local semantic boilerplate filtering inside candidate chunks

Global boilerplate detection combines:

- TF-IDF salience
- semantic centrality using sentence embeddings
- cross-section repetition

Then chunking uses adaptive chunk size derived from sentence statistics.

Why this matters:

- support documents often repeat disclaimers, headers, footers, policy fragments, and UI junk
- if you index that noise, retrieval quality degrades badly
- removing repeated boilerplate improves cluster coherence and answer grounding

### 4.4 Website-specific ingestion path

Website content is handled differently from PDFs.

Why:

- website text is often noisier
- nav labels, footers, cookies, and path-like fragments should not dominate retrieval
- website content benefits more from lexical retrieval and sentence-cleaning than from aggressive section-based clustering

Design choice:

- website content is cleaned and chunked separately
- website chunks are not clustered with HDBSCAN
- they are stored in a separate website vector store and also indexed into Elasticsearch

This is a very defendable design in interview:

"I separated website chunks from PDF clusters because website data behaves differently. That reduced cluster pollution and let me tune retrieval behavior differently for structured docs vs dynamic site content."

### 4.5 Embedding model

Embedding model:

- `SentenceTransformer("all-MiniLM-L6-v2")`

Used for:

- chunk embeddings
- experience memory embeddings
- query embeddings
- semantic dedupe in autocomplete

Why I used it:

- strong quality-to-latency tradeoff
- lightweight enough for CPU serving
- 384-dimensional embeddings work well with FAISS inner-product similarity
- common and defendable production baseline

Tradeoff:

- not domain-specialized
- could miss specialized vocabulary or multilingual nuance

### 4.6 Clustering architecture

Main file:

- `app/pipeline/clustering_pipeline.py`

Architecture:

1. embed document chunks
2. reduce embedding space with UMAP
3. run HDBSCAN across a parameter grid
4. score cluster quality
5. pick best config
6. label clusters
7. build hierarchical FAISS index

Why UMAP + HDBSCAN:

- UMAP preserves neighborhood structure while reducing noisy high-dimensional geometry
- HDBSCAN handles varying density and naturally identifies noise
- support documentation often has uneven topic density, so density-based clustering is a better fit than forced k-means

How I would explain it:

"I wanted topic routing without forcing every chunk into a fixed number of clusters. UMAP compresses the embedding manifold, then HDBSCAN finds dense semantic regions and lets outliers remain noise. That matches real support documents, where some topics are dense and some chunks are miscellaneous."

### 4.7 Hierarchical retrieval architecture

Main file:

- `app/pipeline/hierarchical_index.py`

Architecture:

1. cluster-level FAISS index on cluster centroids
2. per-cluster FAISS index on chunk embeddings

Query-time flow:

1. route query to top clusters
2. search only chunks within those clusters
3. fuse semantic scores with Elasticsearch BM25 when enabled

Why this is useful:

- reduces search space
- gives topic routing before chunk retrieval
- supports interpretable cluster-level analytics
- makes hybrid retrieval more targeted by restricting BM25 to routed clusters

### 4.8 Online answer flow

Main endpoint:

- `POST /answer`

Orchestration:

- built with LangGraph in `app/graph/build_graph.py`
- nodes implemented in `app/graph/nodes.py`

Graph stages:

1. intent classification
2. feedback-state check from experience memory
3. semantic memory lookup for follow-up context
4. primary or secondary retrieval
5. tool retrieval for social/web freshness when needed
6. analyzer decides confidence and escalation
7. owner/human resolution
8. final answer generation

This is a strong uniqueness point:

"Instead of one straight RAG chain, I built an answer graph. That lets the system switch behavior for dissatisfaction retries, website lookups, latest social updates, memory follow-ups, and low-confidence situations."

### 4.9 Experience memory

Main files:

- `app/pipeline/vector_store.py`
- `server/controllers/chatController.js`
- `server/models/Experience.js`
- `server/models/Escalation.js`

What it does:

- stores embeddings of past question experiences
- reuses good past answers if the user asks a semantically similar question
- updates memory state with feedback and negative counts
- escalates repeated failure patterns to humans

Why this is important:

- not all learning has to wait for formal retraining
- experience memory provides near-online adaptation
- semantic groups let feedback propagate across similar questions

Interview framing:

"I added a feedback-aware experience memory layer so the system can reuse or avoid past answers before going back through full RAG. That gives a practical middle layer between stateless generation and full model retraining."

### 4.10 Autocomplete model architecture

Main file:

- `app/autocomplete_training_pipeline.py`

This is a real custom DL component, not just Elastic suggest.

Architecture:

1. SentencePiece unigram tokenizer
2. token embeddings
3. character encoder using embedding plus Conv1D plus max pooling
4. token embedding plus char embedding fusion
5. positional embedding
6. lightweight transformer decoder-style stack with causal mask
7. next-token prediction head

It also uses:

- bot-specific question logs
- semantic dedupe of user questions
- seed question bootstrap from docs and website content
- recency-weighted training
- augmentation with polite prefixes and typo variants
- hybrid suggestion serving with model plus Elasticsearch

Why I would defend this:

"I wanted autocomplete to reflect each bot's domain instead of generic search-box completion. So I built a bot-specific lightweight language model over historical questions, then fused it with Elasticsearch for lexical recall and popularity. The model gives generative completion behavior, while Elasticsearch keeps recall and stability high."

## 5. Elasticsearch in Production

Main file:

- `app/pipeline/elasticsearch_hybrid.py`

### 5.1 How Elasticsearch is used

There are two logical indexes:

1. chunk index
   Stores bot chunk text, topic, source type, source URL, PDF name, and metadata.

2. question index
   Stores canonical autocomplete questions, completion suggester payload, ask count, source, and timestamps.

### 5.2 Why Elasticsearch is important here

It is used for two different jobs:

1. hybrid retrieval for RAG
   Semantic retrieval alone can miss exact names, policy phrases, URLs, contact info, or rare keywords.

2. autocomplete lexical recall
   The custom autocomplete model is useful, but Elastic gives strong prefix and popularity-based recall.

### 5.3 Hybrid retrieval design

For document chunks:

- FAISS semantic retrieval gives meaning-based relevance
- Elasticsearch BM25 gives lexical precision
- final ranking uses weighted fusion

Why this is good:

- semantic search handles paraphrases
- BM25 handles exact entities and sparse keyword matches
- the combination is much more production-safe than either alone

### 5.4 Production resilience built into the ES layer

The ES client includes:

- enable/disable flag
- request timeout and ping timeout
- failure cooldown
- lazy client init
- fallback behavior if ES is unavailable
- strict tenant and bot isolation in query filters

This is a subtle but important design point:

"I treated Elasticsearch as an optional accelerator, not a single point of total failure. If ES is unhealthy, the system degrades to semantic retrieval instead of fully breaking."

### 5.5 Problems we might face with Elasticsearch in production

1. Mapping drift
   Changing fields in app code can break queries or require reindexing.

2. Multi-tenant leakage risk
   If filters are ever omitted, search could leak data across bots or clients.

3. Query latency spikes
   Poor queries, large indexes, or expensive wildcard-like behavior can slow responses.

4. Reindex cost
   Full replace flows can be expensive for large bot corpora.

5. Ranking mismatch
   BM25 scores and semantic scores are not naturally comparable, so fusion weights need tuning.

6. Completion suggester limitations
   Completion is great for prefixes but weaker for typo-heavy or semantic autocomplete alone.

7. External dependency risk
   Managed Elastic outages or auth issues degrade hybrid quality.

8. Consistency lag
   New chunks/questions may not be immediately reflected depending on refresh behavior.

## 6. MLOps Stack in Depth

### 6.1 DVC

Files:

- `dvc.yaml`
- `params.yaml`
- `app/pipeline/dvc_auto.py`
- `scripts/dvc/run_cluster_train.py`
- `scripts/dvc/track_autocomplete_artifacts.py`

What DVC does here:

- versions bot-specific clustering artifacts
- tracks autocomplete artifacts
- uses `params.yaml` to pass `client_id`, `bot_id`, `rebuild_mode`, and manifest path
- can push artifacts to S3

Why I used it:

- reproducibility of pipeline outputs
- artifact versioning outside plain Git
- clean separation between code versioning and heavy model/index artifacts

Important nuance:

This is not only offline MLOps. DVC is triggered from runtime flows inside the Python service.

### 6.2 MLflow

Files:

- `app/pipeline/mlflow_tracking.py`
- `infra/airflow_mlflow_stack/mlflow/Dockerfile`

What MLflow tracks:

- clustering run params and metrics
- autocomplete run params and metrics
- training duration
- model version
- artifacts like model files and clustering summaries

Why I used it:

- experiment visibility
- comparing runs across bots or time
- storing metrics beyond local logs
- artifacts linked to each run

Production setup:

- tracking server in Docker
- Postgres backend store
- S3 artifact destination

### 6.3 Airflow

Files:

- `app/pipeline/airflow_trigger.py`
- `airflow/dags/hdbscan_pipeline_dag.py`
- `airflow/dags/autocomplete_pipeline_dag.py`

What Airflow does:

- orchestrates HDBSCAN pipeline trigger and deploy hook
- orchestrates autocomplete retraining and deploy hook
- receives DAG run triggers from the Python service over API

Why I used it:

- moves retraining orchestration out of request threads
- provides replayability and operational visibility
- cleaner separation between app serving and training control

### 6.4 Monitoring

File:

- `app/pipeline/model_monitoring.py`

What is monitored:

- HDBSCAN retrieval drift-like signals
  - uncategorized queries
  - semantic empty queries
- autocomplete quality signals
  - empty suggestions
  - low-confidence suggestions
  - average max confidence

Why this matters:

"I wanted lightweight operational monitoring even before a full observability stack. So I log behavior-based drift proxies that tell me when retrieval or autocomplete quality is degrading."

### 6.5 MLOps problems we might face

1. Runtime training on serving infrastructure
   Training can steal CPU/RAM from inference.

2. Artifact consistency
   Local storage, DVC state, MLflow artifacts, and deployed models can drift apart.

3. Partial failures
   Training may succeed locally but fail during DVC push or Airflow trigger.

4. Reproducibility gaps
   Some parts still depend on live content, environment variables, and container state.

5. Deployment coupling
   Airflow deploy step uses a shell command hook, which is flexible but operationally fragile.

6. Storage growth
   Per-bot indexes, models, and manifests can accumulate quickly.

7. Lack of dedicated feature/data registry
   Data lineage exists, but not with a full feature store or formal model registry workflow.

8. Monitoring simplicity
   Current drift detection is heuristic, not statistically rigorous.

## 7. ML and DL Architectures Used

### 7.1 Used in doc service

- SentenceTransformer `all-MiniLM-L6-v2` for embeddings
- UMAP for dimensionality reduction
- HDBSCAN for unsupervised clustering
- FAISS inner-product indexes for semantic retrieval
- LangGraph for control-flow orchestration
- Gemini LLM for answer generation, intent routing, and bootstrap question generation

### 7.2 Used in autocomplete

- SentencePiece unigram tokenizer
- character-level CNN encoder
- transformer-style autoregressive next-token model
- Elastic completion plus lexical fallback
- semantic dedupe using embedding similarity

### 7.3 Why the architecture mix is unique

This service is not relying on one monolithic model.
It combines:

- classical IR
- unsupervised ML
- vector search
- lightweight custom DL
- graph orchestration
- LLM prompting

That is a strong systems-design story because each component solves a different failure mode.

## 8. What Makes This Project Unique

### 8.1 Unique engineering ideas

1. Boilerplate-aware ingestion instead of naive chunking.
2. Different handling for PDFs vs website content.
3. Hierarchical cluster-routing before chunk search.
4. Hybrid FAISS plus Elasticsearch retrieval.
5. Graph-based answer routing instead of single-chain RAG.
6. Experience memory with feedback propagation and semantic grouping.
7. Human escalation as part of the system design, not as an afterthought.
8. Bot-specific autocomplete model rather than generic lexical suggest only.
9. Runtime MLOps hooks integrated directly into product behavior.

### 8.2 Your product idea behind it

The product idea is that enterprise support bots should:

- know the company's docs
- know the website
- know fresh social updates when needed
- learn from repeated user behavior
- fail safely to humans
- improve both answer quality and question-entry UX over time

That is much stronger than a standard "chat with PDF" demo.

## 9. Strong Interview Answers for "Why Did You Use X?"

### Why Elasticsearch?

"Because semantic retrieval alone is weak for exact phrases, names, URLs, and prefix search. I used Elasticsearch to add lexical precision for both RAG and autocomplete."

### Why HDBSCAN instead of k-means?

"Because support document topics are uneven in density. HDBSCAN handles noise and variable-size clusters better than forcing a fixed cluster count."

### Why UMAP before HDBSCAN?

"To reduce noisy high-dimensional embedding space while preserving neighborhood structure, which improves density-based clustering quality."

### Why FAISS if Elasticsearch exists?

"Elasticsearch handles lexical retrieval well, but FAISS is better for fast semantic nearest-neighbor search. I used both because they solve different retrieval problems."

### Why MLflow?

"To track parameters, metrics, timing, and artifacts for clustering and autocomplete runs so model behavior is inspectable over time."

### Why DVC too?

"MLflow tracks experiments, but DVC versions pipeline artifacts and data-like outputs. I used DVC for reproducible artifact lineage and MLflow for experiment observability."

### Why Airflow?

"Because retraining and deployment orchestration should not live only inside request-handling code. Airflow gives explicit, triggerable, operational workflows."

## 10. Weaknesses I Would Honestly Admit

- The stack is powerful but heavy for a single VM.
- Some stores are still in-memory and process-local, which makes horizontal scaling harder.
- Monitoring uses heuristic drift signals rather than rigorous offline evaluation pipelines.
- Social freshness retrieval depends on web scraping/search behavior, which can be brittle.
- The autocomplete model is custom and interesting, but harder to maintain than a simpler retrieval-only solution.

Admitting these makes you sound stronger, not weaker.

## 11. How To Present This As "Completely Mine"

When explaining the project, always answer in this order:

1. problem I wanted to solve
2. why naive RAG was not enough
3. the architecture choices I made
4. tradeoffs and failure modes
5. what I would improve next

Use the phrase:

"I designed it this way because..."

That keeps the explanation ownership-driven instead of feature-list-driven.

## 12. Final One-Line Identity Statement

"This project is my attempt to turn a support chatbot from a simple RAG demo into a production-style learning system that combines retrieval, memory, safety, freshness, and MLOps discipline."
