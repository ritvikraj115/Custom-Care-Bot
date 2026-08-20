# Custom Care Bot

Custom Care Bot is a multi-tenant AI customer support platform. A business can create a bot, upload PDFs, connect website and social context, let customers ask questions, collect feedback, retry weak answers, escalate repeated failures to a human, and improve future answers using experience memory and MLOps pipelines.

This README is written as a simple end-to-end interview guide. If you understand this file, you should be able to explain the product, architecture, data flow, models, algorithms, tradeoffs, deployment, and future improvements confidently.

## 1. One-Minute Explanation

Custom Care Bot solves the problem of support teams repeatedly answering the same customer questions while ordinary chatbots give stale or hallucinated answers.

The system lets every company create its own support bot. Each bot has its own documents, website data, social data, chat sessions, feedback, analytics, and learned question patterns. When a customer asks a question, the system does not blindly ask an LLM. It first retrieves relevant business content, checks previous experiences, routes the request through a LangGraph workflow, and then uses Gemini to produce a grounded answer. If the system is not confident, or if users keep giving negative feedback, it escalates the question to the business owner.

In short:

```text
Business documents + website + social context
        -> retrieval and graph routing
        -> grounded customer answer
        -> feedback, retry, escalation, analytics
        -> better future answers
```

## 2. Business Problem

Customer support teams face five common problems:

| Problem | Real-world impact | How this project handles it |
| --- | --- | --- |
| Repetitive questions | Agents waste time answering FAQs manually | Customers ask the bot first |
| Stale knowledge | Answers become outdated when policies or websites change | Bots ingest PDFs, website content, and social context |
| Hallucination | Generic LLMs may invent answers | Answers are grounded in retrieved business content |
| No learning loop | Bad chatbot answers repeat forever | Feedback, retry, semantic memory, and escalation are stored |
| No operational visibility | Owners cannot see what the bot is failing on | Analytics expose sessions, messages, negative feedback, unresolved questions, and escalations |

The product is useful for SaaS support teams, ecommerce stores, educational institutes, HR and onboarding teams, internal knowledge bases, clinics, service businesses, finance support, banking support, and insurance support.

## 3. Product Features

| Feature | What it means |
| --- | --- |
| Multi-tenant accounts | Each client/company has isolated bots and data |
| Bot creation | A client can create multiple bots for different purposes |
| PDF upload | Owners upload business documents for RAG |
| Website ingestion | The Python service can crawl and index website content |
| Social refresh | Social links can be fetched and indexed for latest public updates |
| Public chat | Customers chat with a bot using a session ID |
| Markdown answers | Bot replies support structured text and lists |
| References | Answers can include document, website, social, or memory references |
| Confidence metadata | Frontend shows confidence and source type |
| Autocomplete | Suggestions adapt to bot-specific user questions |
| Feedback | Users can mark answers useful or not useful |
| Retry | Negative feedback can trigger secondary retrieval |
| Escalation | Repeated failures create owner-visible escalations |
| Human resolution | Owner answer becomes trusted memory for future similar questions |
| Analytics | Dashboard summarizes usage, feedback, weak areas, and open issues |
| MLOps | DVC, MLflow, and optional Airflow support training and tracking |

## 4. High-Level Architecture

```text
React frontend
    |
    v
Node/Express API
    |
    |-- MongoDB
    |     - clients
    |     - bots
    |     - documents
    |     - chat sessions
    |     - chat messages
    |     - experiences
    |     - escalations
    |
    v
FastAPI document intelligence service
    |
    |-- PDF and website ingestion
    |-- chunking and embeddings
    |-- FAISS vector indexes
    |-- optional Elasticsearch BM25 retrieval
    |-- LangGraph answer routing
    |-- Gemini LLM generation
    |-- autocomplete model training
    |-- DVC / MLflow / Airflow hooks
```

The project is split into three main services because each service has a different responsibility:

| Service | Folder | Main responsibility |
| --- | --- | --- |
| Frontend | `chatbot/` | UI for owners and chat users |
| Backend API | `server/` | Authentication, tenants, bots, documents, chat, feedback, analytics |
| AI/RAG service | `python_doc_service/` | Ingestion, retrieval, ML models, graph routing, answer generation |

## 5. Why Three Services?

The separation is intentional.

| Decision | Why |
| --- | --- |
| React frontend | Fast UI development, simple routing, easy deployment as static assets |
| Node/Express backend | Good fit for product APIs, auth, file upload orchestration, MongoDB models |
| Python AI service | Python has stronger ML/RAG libraries: sentence-transformers, FAISS, UMAP, HDBSCAN, TensorFlow, LangGraph |

Why not put everything in one backend?

- The ML pipeline needs Python libraries that are awkward in Node.
- Long document processing should not block the product API.
- AI service can scale separately from the normal web API.
- Business state stays in Node while intelligence pipelines stay in Python.

## 6. Repository Map

```text
Custom-Care-Bot/
  chatbot/                         React frontend
  server/                          Express backend
  python_doc_service/              FastAPI AI/RAG service
  infra/airflow_mlflow_stack/       Docker Compose stack for Airflow + MLflow
  scripts/windows/                 Windows deployment/helper scripts
  AZURE_CICD.md                    Azure CI/CD guide
  PRODUCTION_DEPLOYMENT.md         Azure production deployment guide
  AWS_END_TO_END_DEVOPS_DEPLOYMENT.md
  MLOPS_DEPLOYMENT.md
  PYTHON_DOC_SERVICE_MLOPS_INTERVIEW_GUIDE.md
  README.md
```

Important code locations:

| File/folder | Purpose |
| --- | --- |
| `chatbot/src/App.js` | Frontend routes and protected pages |
| `chatbot/src/api/axios.js` | Axios client and auth header injection |
| `chatbot/src/pages/Dashboard.jsx` | Owner dashboard and bot list |
| `chatbot/src/pages/BotDetail.jsx` | Upload documents, view analytics, manage escalations |
| `chatbot/src/components/ChatWindow.jsx` | Chat UI, autocomplete, feedback, retry, trace display |
| `server/server.js` | Express app, database connection, route mounting |
| `server/routes/auth.js` | Register/login |
| `server/routes/bots.js` | Bot CRUD and analytics |
| `server/routes/documents.js` | Upload documents and track ingestion jobs |
| `server/controllers/chatController.js` | Chat sessions, answers, feedback, retry, escalation |
| `server/models/` | MongoDB schemas |
| `python_doc_service/app/main.py` | FastAPI endpoints |
| `python_doc_service/app/pipeline/` | Extraction, chunking, embeddings, clustering, indexes |
| `python_doc_service/app/graph/` | LangGraph answer workflow |
| `python_doc_service/app/autocomplete_training_pipeline.py` | Bot-specific autocomplete model |
| `python_doc_service/dvc.yaml` | DVC pipeline stages |
| `python_doc_service/params.yaml` | DVC runtime parameters |

## 7. Tech Stack

### Frontend

| Technology | Usage |
| --- | --- |
| React 19 | UI framework |
| React Router | Page routing |
| Axios | API requests |
| react-markdown | Render assistant answers |
| remark-gfm | GitHub-flavored markdown support |
| React Toastify | User notifications |

### Backend

| Technology | Usage |
| --- | --- |
| Node.js | Runtime |
| Express 5 | REST API |
| MongoDB | Database |
| Mongoose | ODM and schema modeling |
| JWT | Authentication |
| bcryptjs | Password hashing |
| multer | PDF upload handling |
| axios/form-data | Calls from backend to Python service |

### Python AI Service

| Technology | Usage |
| --- | --- |
| FastAPI | AI/RAG API |
| PyMuPDF | PDF text extraction |
| spaCy | Sentence splitting and text processing |
| SentenceTransformers | Embeddings |
| `all-MiniLM-L6-v2` | 384-dimensional embedding model |
| FAISS | Vector search |
| UMAP | Dimensionality reduction before clustering |
| HDBSCAN | Density-based clustering |
| scikit-learn TF-IDF | Boilerplate detection and cluster labels |
| Elasticsearch | Optional BM25/hybrid retrieval |
| LangGraph | Answer workflow orchestration |
| Gemini | LLM answer generation |
| TensorFlow/Keras | Autocomplete model |
| SentencePiece | Autocomplete tokenizer |
| DVC | Pipeline and artifact tracking |
| MLflow | Experiment metrics and model tracking |
| Airflow | Optional scheduled/triggered ML workflows |
| Playwright/trafilatura | Website/social extraction |

## 8. Main User Journeys

### 8.1 Business Owner Registration

```text
Owner fills registration form
    -> React calls POST /api/auth/register
    -> Express hashes password with bcrypt
    -> Client is saved in MongoDB
```

The system stores company name, email, password hash, industry, and timestamps. It does not store the raw password.

### 8.2 Login

```text
Owner submits email and password
    -> Express validates password
    -> Express signs JWT with clientId
    -> Frontend stores token in localStorage
    -> Axios adds Authorization header on future requests
```

JWT payload:

```json
{
  "clientId": "mongo_client_id"
}
```

Token expiration:

```text
1 day
```

### 8.3 Bot Creation

```text
Owner creates bot
    -> POST /api/bots
    -> Bot is linked to owner tenantId
    -> If website/social URLs exist, backend queues background social refresh
```

Bot fields:

- bot name
- purpose
- description
- website URL
- Facebook URL
- Instagram URL
- public access flag
- active flag

### 8.4 Document Upload And Ingestion

```text
Owner uploads PDFs
    -> React sends multipart form data to Express
    -> Express verifies bot ownership
    -> Files are hashed with SHA-256
    -> Duplicate documents are skipped
    -> Document metadata is saved in MongoDB
    -> Ingestion job is queued
    -> Express sends files to FastAPI /process
    -> Python extracts, chunks, embeds, clusters, indexes
    -> Frontend polls job status
```

The backend uses an in-memory ingestion queue. This keeps upload response fast, but it is not durable across process restarts. In production, this should move to Redis, BullMQ, SQS, RabbitMQ, or another real job queue.

### 8.5 Customer Chat

```text
Customer opens chat page
    -> POST /api/chat/session/:botId
    -> Express creates ChatSession
    -> Python returns starter questions
    -> Customer asks question
    -> Express stores user message
    -> Express calls FastAPI /answer
    -> LangGraph retrieves context and builds answer
    -> Express stores assistant message and Experience
    -> Frontend displays answer, confidence, references, trace, feedback buttons
```

### 8.6 Feedback, Retry, And Escalation

```text
User dislikes answer
    -> POST /api/chat/feedback
    -> Experience feedbackScore decreases
    -> Similar semantic questions are grouped
    -> If retry is still allowed, user can retry
    -> Retry uses secondary retrieval and excludes previous weak chunks
    -> If still poor, escalation is created
    -> Owner resolves escalation with trusted answer
    -> Trusted answer is indexed as owner memory
```

This is the main learning loop of the product.

## 9. Data Flow Diagrams

### 9.1 Upload Pipeline

```text
PDF upload
  -> Express multer
  -> SHA-256 duplicate check
  -> MongoDB Document records
  -> In-memory ingestion job
  -> FastAPI /process
  -> PyMuPDF extraction
  -> heading and section detection
  -> boilerplate removal
  -> semantic chunking
  -> SentenceTransformer embeddings
  -> UMAP dimensionality reduction
  -> HDBSCAN clustering
  -> FAISS hierarchical index
  -> optional Elasticsearch sync
  -> DVC/MLflow tracking
```

### 9.2 Answer Pipeline

```text
User question
  -> Express chat controller
  -> conversation history
  -> experience memory search
  -> FastAPI /answer
  -> LangGraph intent classifier
  -> feedback state check
  -> semantic memory lookup
  -> primary retrieval
  -> optional secondary/tool retrieval
  -> analyzer
  -> Gemini grounded answer
  -> references and confidence
  -> MongoDB chat message + experience
```

### 9.3 Human Escalation Pipeline

```text
Repeated negative feedback
  -> semantic group feedback drops
  -> escalation is created
  -> owner sees open escalation
  -> owner writes correct answer
  -> owner answer stored as Experience
  -> answer indexed in vector memory
  -> future similar questions reuse trusted answer
```

## 10. Frontend Design

The frontend is intentionally simple:

- owner logs in
- owner sees bot dashboard
- owner creates bots
- owner uploads documents
- owner sees analytics and escalations
- user chats with a bot

Routes in `chatbot/src/App.js`:

| Route | Component | Purpose |
| --- | --- | --- |
| `/login` | `Login` | Owner login |
| `/register` | `Register` | Owner registration |
| `/` | `Dashboard` | Protected owner dashboard |
| `/create-bot` | `CreateBot` | Protected bot creation |
| `/bots/:botId` | `BotDetail` | Protected bot detail, uploads, analytics, escalations |
| `/chat/:botId` | `PublicChat` | Chat UI for a bot |

Important frontend behavior:

- Axios base URL comes from `REACT_APP_API_BASE_URL`.
- JWT is read from `localStorage`.
- Axios injects `Authorization: Bearer <token>`.
- Chat answers are rendered as markdown.
- Chat suggestions are cached briefly on the client.
- Feedback and retry are available per assistant answer.
- The UI shows confidence, source type, references, and analysis trace.

Interview note:

The backend chat session endpoint supports public bot sessions, but the current frontend route wraps `/chat/:botId` in `ProtectedRoute`. If the product is meant to expose public customer chat links, that route should be made public in the frontend while keeping owner-only routes protected.

## 11. Backend Design

The backend is the product control plane.

It owns:

- authentication
- tenant isolation
- bot ownership
- document metadata
- upload orchestration
- chat persistence
- feedback
- retry coordination
- escalations
- analytics

It does not do heavy ML work. Instead, it calls the Python service.

### 11.1 Backend Startup

`server/server.js` does the following:

```text
load .env
create Express app
enable CORS
enable JSON body parsing
disable ETag
connect MongoDB
mount API routes
start social refresh scheduler
listen on PORT
```

Health endpoint:

```text
GET /api/health
```

### 11.2 Authentication

Files:

- `server/routes/auth.js`
- `server/middleware/authMiddleware.js`

Register:

```text
POST /api/auth/register
```

Login:

```text
POST /api/auth/login
```

Security choices:

| Choice | Reason |
| --- | --- |
| bcrypt password hashing | Raw passwords should never be stored |
| JWT auth | Stateless API authentication |
| `clientId` in token | Every protected request can be scoped to one tenant |
| Bearer token header | Standard API auth pattern |

### 11.3 Tenant Isolation

Most owner APIs use `req.clientId` from the JWT. For example:

- list only bots where `tenantId = req.clientId`
- upload only to a bot owned by `req.clientId`
- view analytics only for owned bots
- resolve only escalations for owned bots

This is critical because the app is multi-tenant.

### 11.4 Document Upload Queue

File:

```text
server/services/ingestionJobService.js
```

The queue stores jobs in memory:

```text
QUEUED -> PROCESSING -> COMPLETED
                  |
                  -> FAILED
```

Why this exists:

- PDF ingestion can take a long time.
- The upload API should return quickly.
- The frontend can poll status.

Tradeoff:

- In-memory queue is simple for a prototype.
- It is not safe for production restarts or multiple backend replicas.
- A production system should use a persistent queue.

### 11.5 Analytics

The analytics endpoint computes:

- total sessions
- total messages
- user messages
- assistant messages
- total experiences
- escalated experiences
- negative feedback count
- retry count
- owner-resolved count
- likely no-document matches
- open and resolved escalations
- unresolved questions
- negative hotspots

Endpoint:

```text
GET /api/bots/:id/analytics
```

This gives the owner a practical support operations dashboard.

## 12. MongoDB Data Model

### 12.1 Client

Represents one company/user account.

Important fields:

| Field | Meaning |
| --- | --- |
| `companyName` | Business name |
| `email` | Login email, unique |
| `passwordHash` | bcrypt hash |
| `industry` | Business category |

### 12.2 Bot

Represents one chatbot owned by one client.

Important fields:

| Field | Meaning |
| --- | --- |
| `tenantId` | Owner client ID |
| `name` | Bot name |
| `description` | Bot description |
| `websiteUrl` | Website source |
| `facebookUrl` | Social source |
| `instagramUrl` | Social source |
| `purpose` | Support, sales, HR, finance, etc. |
| `publicAccess` | Whether public users may access it |
| `isActive` | Whether bot is active |

Interview note:

In the current code, some Mongoose refs use `"User"` even though the actual owner model is `Client`. Functionally the ObjectId still stores correctly, but the ref name should be changed to `Client` for cleaner population and maintainability.

### 12.3 Document

Represents an uploaded PDF.

Important fields:

| Field | Meaning |
| --- | --- |
| `clientId` | Owner |
| `botId` | Bot |
| `fileName` | Original file name |
| `filePath` | Local upload path |
| `fileSize` | Size in bytes |
| `contentHash` | SHA-256 hash for deduplication |

Important index:

```text
clientId + botId + contentHash
```

This prevents duplicate uploads from rebuilding the same content.

### 12.4 ChatSession

Represents one user conversation with a bot.

Important fields:

- bot ID
- tenant ID
- source
- hashed IP
- user agent
- last active timestamp

The IP is hashed to avoid storing raw IP directly.

### 12.5 ChatMessage

Stores messages inside a session.

Important fields:

- session ID
- role: `user` or `assistant`
- content
- created timestamp

### 12.6 Experience

This is one of the most important collections.

An Experience stores the system's memory of a question-answer event:

| Field | Meaning |
| --- | --- |
| `question` | User question |
| `answer` | Assistant or owner answer |
| `retrievedChunks` | Chunks used during retrieval |
| `avgChunkSimilarity` | Average retrieval similarity |
| `feedbackScore` | User feedback score |
| `negativeCount` | Negative feedback count |
| `semanticId` | Groups similar questions |
| `retrievalVariant` | primary, secondary, or owner |
| `resolvedByOwner` | Whether this is a trusted human answer |
| `status` | active or escalated |

Why it matters:

- prevents repeating known bad answers
- allows retry with different chunks
- stores owner corrections
- powers semantic memory search
- supports analytics and escalation

### 12.7 Escalation

Represents a question that needs human attention.

Important fields:

- bot ID
- question
- related experience IDs
- reason
- status: open or resolved
- resolved answer
- resolved timestamp

## 13. Python AI Service

The Python service is the intelligence engine.

File:

```text
python_doc_service/app/main.py
```

Main endpoint groups:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health |
| `POST /process` | Ingest PDFs and website data |
| `POST /answer` | Generate grounded answer |
| `POST /autocomplete/record-question` | Store user question for autocomplete training |
| `POST /autocomplete/suggest` | Return query suggestions |
| `POST /autocomplete/train` | Train autocomplete model |
| `POST /autocomplete/top-questions` | Return starter questions |
| `POST /experience/index` | Store answer experience in vector memory |
| `POST /experience/search` | Search previous experiences |
| `POST /experience/update-feedback` | Update memory feedback |
| `POST /social/refresh` | Refresh social context |
| `POST /bot/delete` | Delete bot indexes and storage |
| `GET /monitoring/model-dashboard` | Model/quality monitoring |

## 14. Document Ingestion Pipeline

The ingestion pipeline converts raw PDFs and website content into searchable knowledge.

### 14.1 PDF Extraction

File:

```text
python_doc_service/app/pipeline/extract.py
```

The service uses PyMuPDF to extract text blocks from PDFs. It also records layout clues such as font size and block position.

Why layout matters:

- headings often have larger fonts
- headings help split content into sections
- section-level chunking gives better retrieval than random page chunks

### 14.2 Heading Detection

The pipeline marks likely headings using signals such as font size, short text, uppercase text, and position in the document. This is not perfect, but it is practical for many business PDFs.

### 14.3 Section Building

After headings are identified, the pipeline groups following text under those headings.

Why:

- user questions usually map to topics or sections
- section boundaries help avoid mixing unrelated content
- retrieval references become easier to explain

### 14.4 Boilerplate Detection

File:

```text
python_doc_service/app/pipeline/chunk.py
```

Many PDFs contain repeated or low-value text:

- copyright lines
- navigation text
- repeated footers
- generic legal phrases
- contact banners

The pipeline detects boilerplate using:

- TF-IDF
- semantic centrality
- repetition across sections
- local semantic similarity

If boilerplate is embedded and indexed, retrieval may return useless chunks. Removing it improves answer quality.

### 14.5 Adaptive Chunking

The chunk size is adapted from sentence statistics instead of using one fixed size.

Simplified logic:

```text
median sentence length
  -> target words
  -> min words
  -> max sentences
```

The project also uses one-sentence overlap to preserve context between chunks.

Why not just use fixed 500-token chunks?

- business documents have different writing styles
- short FAQ documents need different chunk sizes than long policy documents
- adaptive chunking reduces irrelevant context

### 14.6 Website Chunking

Website text is noisier than PDF text. The pipeline applies extra cleanup for website content:

- removes cookie/privacy/login/follow-us patterns
- removes navigation-like text
- creates compact website chunks
- stores website source metadata separately

### 14.7 Embeddings

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Why this model:

- lightweight
- fast
- good enough semantic retrieval quality
- easy to run locally
- compatible with FAISS cosine-style similarity after normalization

Why not use a very large embedding model?

- slower ingestion
- higher memory cost
- harder local deployment
- unnecessary for a student/prototype support bot platform

### 14.8 UMAP + HDBSCAN Clustering

The pipeline clusters document chunks.

Steps:

```text
embeddings
  -> UMAP dimensionality reduction
  -> HDBSCAN clustering
  -> cluster labels using TF-IDF
  -> hierarchical index
```

Why UMAP:

- high-dimensional embeddings are hard to cluster directly
- UMAP preserves neighborhood structure
- clustering becomes more stable in lower dimensions

Why HDBSCAN:

- does not require choosing number of clusters
- handles noise/outliers
- works well when topics have different densities

Why not K-Means:

- K-Means needs a fixed K
- support documents have unknown topic counts
- K-Means forces every chunk into a cluster, even noise

### 14.9 Clustering Score

The code selects the best UMAP/HDBSCAN parameters using:

```text
score = 0.45 * coherence
      - 0.35 * centroid_similarity
      - 0.20 * noise_ratio
```

Meaning:

| Term | Meaning | Desired |
| --- | --- | --- |
| coherence | chunks inside a cluster are similar | high |
| centroid_similarity | different clusters are too similar | low |
| noise_ratio | chunks marked as noise | low |

This balances three goals:

- clusters should be internally meaningful
- clusters should be different from each other
- not too much content should be thrown away as noise

### 14.10 Cluster Labels

Cluster labels are generated using TF-IDF top terms.

Example:

```text
refund policy shipping
account login reset
pricing subscription invoice
```

These labels are not the final answer. They help organize retrieval and make debugging easier.

## 15. Retrieval Design

The project uses hierarchical retrieval instead of one flat vector search.

### 15.1 Hierarchical FAISS Index

File:

```text
python_doc_service/app/pipeline/hierarchical_index.py
```

Indexes:

```text
cluster centroid index
  -> per-cluster chunk index
```

Retrieval:

```text
query embedding
  -> find most relevant clusters
  -> search chunks inside those clusters
  -> return top chunks
```

Why hierarchical retrieval:

- faster than searching every chunk for large bots
- improves topic-level organization
- reduces unrelated chunk matches

### 15.2 FAISS

FAISS is used for vector similarity search.

The project normalizes embeddings and uses inner product search, which behaves like cosine similarity for normalized vectors.

### 15.3 Elasticsearch Hybrid Retrieval

Elasticsearch is optional but supported.

It adds BM25 lexical search, which helps when user queries contain exact words:

- product names
- policy codes
- acronyms
- pricing terms
- names
- error messages

Hybrid score:

```text
hybrid_score = semantic_weight * vector_score
             + bm25_weight * bm25_score
```

Default weights:

```text
semantic = 0.45
BM25     = 0.55
```

Why hybrid retrieval:

- vector search understands meaning
- BM25 catches exact keywords
- combining both is more reliable than either alone

### 15.4 Source-Aware Retrieval

The retriever can search uploaded documents, website chunks, social context, and owner memory. The graph can prefer website or social tools for queries that look time-sensitive or website-specific.

## 16. LangGraph Answer Workflow

Files:

```text
python_doc_service/app/graph/build_graph.py
python_doc_service/app/graph/nodes.py
python_doc_service/app/graph/retrievers.py
python_doc_service/app/graph/tools.py
```

The answer process is a graph, not a single function call.

Simplified graph:

```text
IntentClassifier
  -> CheckFeedbackState
  -> SemanticMemoryLookup
  -> PrimaryRetrieval
  -> Analyzer
  -> maybe ToolRetrieval
  -> maybe HumanInLoop
  -> maybe OwnerResolution
  -> FinalizeResponse
```

### 16.1 Intent Classification

The graph classifies questions into labels such as:

- `docs_lookup`
- `website_lookup`
- `latest_social_updates`
- `memory_followup`
- `dissatisfied_retry`

Different questions need different context. A pricing page question may need website content. A latest update question may need social context. A repeated negative question may need memory or escalation handling.

### 16.2 Feedback State Check

Before answering, the system checks whether similar questions have received negative feedback before. If users already disliked an answer pattern, the system should avoid repeating it.

### 16.3 Semantic Memory Lookup

The graph searches previous experiences. Owner-resolved answers are preferred because they are trusted human corrections.

### 16.4 Primary Retrieval

Primary retrieval searches the normal document/website/social indexes.

The default backend request asks for:

```text
top_k = 3
```

### 16.5 Secondary Retrieval

Secondary retrieval is used during retry. It excludes chunks that contributed to the weak answer and searches wider.

If the first retrieved context was poor, asking the same retrieval path again usually returns the same bad context. Secondary retrieval forces the system to look elsewhere.

### 16.6 Analyzer

The analyzer decides whether the retrieved evidence is good enough.

It looks at:

- retrieval confidence
- average similarity
- negative feedback history
- source type
- no-docs state
- tool results

If evidence is weak, the system can route to human escalation instead of hallucinating.

### 16.7 Final Answer

Gemini is instructed to answer using only:

- retrieved docs
- tools
- memory

If the context is insufficient, the prompt tells it to say that verified information is not available. This is a key anti-hallucination design.

## 17. LLM Usage

LLM provider:

```text
Google Gemini
```

Default model from environment:

```text
GEMINI_MODEL=gemini-2.5-flash
```

Default temperature:

```text
0.2
```

Why low temperature:

- support answers should be factual
- creativity is less important than consistency
- low temperature reduces variation

What the LLM does:

- intent classification
- starter question generation
- final answer generation
- fallback reasoning when needed

What the LLM should not do:

- invent company policy
- answer from general world knowledge when business content is missing
- override owner-resolved answers

## 18. Experience Memory

Experience memory is how the system learns from past conversations.

File:

```text
python_doc_service/app/pipeline/vector_store.py
```

When an answer is produced, the backend creates an `Experience` in MongoDB and indexes its text in the Python vector store.

Experience ranking:

```text
rank_score = similarity + 0.15 * feedback_score
```

Meaning:

- semantically similar experiences rank higher
- positive feedback improves rank
- negative feedback lowers rank
- owner answers get priority

Minimum similarity threshold:

```text
EXPERIENCE_MIN_SIMILARITY=0.72
```

The value is clamped between:

```text
0.55 and 0.95
```

Why experience memory matters:

- remembers what worked
- avoids what failed
- groups similar questions using `semanticId`
- supports human correction
- makes the bot improve without retraining the whole document index

## 19. Feedback Logic

Important backend constants:

```text
FEEDBACK_BLOCK_THRESHOLD  = -2
RETRY_SCORE_FLOOR        = -1
ESCALATION_SCORE_FLOOR   = 0
SECONDARY_MIN_SIMILARITY = 0.15
OWNER_RESOLVED_SCORE     = 5
```

Simple meaning:

| Constant | Meaning |
| --- | --- |
| `RETRY_SCORE_FLOOR` | allow retry while feedback is not too bad |
| `FEEDBACK_BLOCK_THRESHOLD` | stop repeating heavily disliked answers |
| `SECONDARY_MIN_SIMILARITY` | retry answer must have enough evidence |
| `OWNER_RESOLVED_SCORE` | human answer starts with high trust |

Feedback values:

```text
positive -> feedbackScore + 1
negative -> feedbackScore - 1
neutral  -> no strong direction
```

If negative feedback repeats, an escalation is created.

## 20. Autocomplete System

Autocomplete is bot-specific. This is important because two companies may have very different customer questions.

Files:

```text
python_doc_service/app/autocomplete_training_pipeline.py
server/controllers/chatController.js
chatbot/src/components/ChatWindow.jsx
```

### 20.1 Data Sources

Autocomplete learns from:

- starter questions generated from documents
- website samples
- actual user questions
- recent user behavior
- Elasticsearch question suggestions if enabled

### 20.2 Model Architecture

The autocomplete model uses:

- SentencePiece unigram tokenizer
- token embeddings
- character-level encoder
- positional embeddings
- causal self-attention
- transformer-style decoder blocks
- Keras/TensorFlow

Simplified architecture:

```text
token input
char input
  -> token embedding
  -> character CNN encoder
  -> positional embedding
  -> transformer decoder blocks
  -> next-token prediction
  -> suggestions
```

Default design details:

| Setting | Value |
| --- | --- |
| embedding/model dimension | 128 |
| max token sequence length | 24 |
| max character length | 15 |
| attention heads | 4 |
| feed-forward size | 256 then 128 |
| dropout | 0.1 |
| vocabulary size | about 240 |

### 20.3 Why Character + Token Model?

Token-level modeling helps with normal phrases.

Character-level encoding helps with:

- partial words
- spelling variations
- product codes
- names
- incomplete user input

### 20.4 Why Bot-Specific Autocomplete?

Generic autocomplete may suggest irrelevant questions. Bot-specific autocomplete suggests questions based on that bot's actual documents and users.

Example:

```text
Banking bot:   how do I reset my debit card pin
HR bot:        how do I apply for leave
Ecommerce bot: what is the refund policy
```

### 20.5 Training Trigger

The system can retrain after enough new user questions.

Default idea in code:

```text
train after around 25 new user questions
```

### 20.6 Frontend Behavior

The chat UI:

- sends autocomplete request after the user types enough characters
- limits suggestions to 3
- asks for up to 3 future words
- caches suggestions briefly to reduce network calls

## 21. Social And Website Context

Website and social content solve a different problem from PDF uploads.

PDFs are usually stable policy documents. Websites and social pages can contain more recent public information.

The Python service can:

- crawl website content
- extract useful text
- convert website content into chunks
- refresh social link content
- store social context by bot
- retrieve social context for latest-update style questions

Important tradeoff:

Social scraping can be brittle because platforms change layouts and restrict access. The code keeps cached social context and disables live fallback by default unless configured.

## 22. API Reference

### 22.1 Auth

```text
POST /api/auth/register
POST /api/auth/login
```

Register body:

```json
{
  "companyName": "Acme",
  "email": "owner@example.com",
  "password": "secret",
  "industry": "Ecommerce"
}
```

Login response:

```json
{
  "token": "jwt_token"
}
```

### 22.2 Bots

```text
GET    /api/bots
POST   /api/bots
GET    /api/bots/:id/analytics
DELETE /api/bots/:id
```

Create body:

```json
{
  "botName": "Acme Support",
  "botPurpose": "Customer Support",
  "description": "Answers product and policy questions",
  "websiteUrl": "https://example.com",
  "facebookUrl": "",
  "instagramUrl": ""
}
```

### 22.3 Documents

```text
GET  /api/documents/bot/:botId
POST /api/documents/upload
GET  /api/documents/jobs/:jobId
```

Upload fields:

```text
botId
files[]
```

Typical upload response:

```json
{
  "status": "QUEUED",
  "rebuildMode": "incremental",
  "jobId": "uuid",
  "skippedDuplicates": [],
  "documents": []
}
```

### 22.4 Chat

```text
POST /api/chat/session/:botId
POST /api/chat/message
GET  /api/chat/history/:sessionId
GET  /api/chat/autocomplete
POST /api/chat/feedback
POST /api/chat/retry
GET  /api/chat/stream
```

Message body:

```json
{
  "sessionId": "mongo_session_id",
  "message": "What is your refund policy?"
}
```

Answer response includes:

```json
{
  "reply": "Grounded answer",
  "experienceId": "mongo_experience_id",
  "confidence": 0.82,
  "sourceType": "docs",
  "references": [],
  "retryAvailable": true,
  "analysis": []
}
```

### 22.5 Escalations

```text
GET  /api/escalations/bot/:botId
POST /api/escalations/resolve
```

Resolve body:

```json
{
  "escalationId": "mongo_escalation_id",
  "answer": "Correct human-approved answer"
}
```

### 22.6 Python Service

```text
GET  /health
POST /process
POST /answer
POST /autocomplete/suggest
POST /autocomplete/train
POST /autocomplete/top-questions
POST /experience/index
POST /experience/search
POST /social/refresh
POST /bot/delete
```

## 23. Environment Variables

### 23.1 Backend

File:

```text
server/.env.example
```

Important variables:

| Variable | Purpose |
| --- | --- |
| `PORT` | Express port |
| `MONGO_URI` | MongoDB connection string |
| `JWT_SECRET` | JWT signing secret |
| `CORS_ORIGIN` | Allowed frontend origin |
| `DOC_SERVICE_BASE_URL` | FastAPI service URL |
| `RAG_TIMEOUT_MS` | Timeout for RAG answer call |
| `DEBUG_RAG_PAYLOAD` | Debug logging flag |
| `SOCIAL_REFRESH_INTERVAL_MS` | Social refresh interval |
| `SOCIAL_REFRESH_BATCH_LIMIT` | Max bots per refresh batch |

### 23.2 Frontend

File:

```text
chatbot/.env.example
```

Variable:

```text
REACT_APP_API_BASE_URL=http://localhost:5000/api
```

### 23.3 Python

File:

```text
python_doc_service/.env.example
```

Important variables:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_MODEL` | LLM model |
| `GEMINI_TEMPERATURE` | Generation randomness |
| `SCRAPED_PDF_DIR` | Temporary PDF storage |
| `RUN_QUALITY_CHECKS` | Optional pipeline validation |
| `ES_ENABLED` | Enable Elasticsearch |
| `ES_URLS` | Elasticsearch URL |
| `ES_CHUNK_INDEX` | Chunk index name |
| `ES_QUESTION_INDEX` | Autocomplete question index |
| `HYBRID_SEMANTIC_WEIGHT` | Vector score weight |
| `HYBRID_BM25_WEIGHT` | BM25 score weight |
| `MLFLOW_ENABLED` | Enable MLflow logging |
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `DVC_AUTO_FLOW_ENABLED` | Enable DVC automation |
| `DVC_AUTO_PUSH` | Push artifacts to remote |
| `AIRFLOW_ENABLED` | Enable Airflow DAG triggering |
| `EXPERIENCE_MIN_SIMILARITY` | Semantic memory match threshold |

## 24. Local Setup

### 24.1 Backend

```bash
cd server
npm install
npm run dev
```

Backend URL:

```text
http://localhost:5000
```

### 24.2 Python Service

```bash
cd python_doc_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Python docs:

```text
http://localhost:8000/docs
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 24.3 Frontend

```bash
cd chatbot
npm install
npm start
```

Frontend URL:

```text
http://localhost:3000
```

## 25. Deployment Options

The repo contains multiple deployment guides:

| File | Purpose |
| --- | --- |
| `PRODUCTION_DEPLOYMENT.md` | Azure managed production deployment |
| `AZURE_CICD.md` | GitHub Actions and Azure CI/CD |
| `AWS_END_TO_END_DEVOPS_DEPLOYMENT.md` | AWS style end-to-end deployment |
| `MLOPS_DEPLOYMENT.md` | MLOps deployment roadmap |
| `infra/airflow_mlflow_stack/README.md` | Docker Compose stack with Airflow and MLflow |

Common production shape:

```text
Static frontend
  -> API service
  -> MongoDB Atlas or managed MongoDB
  -> Python AI service
  -> object storage for uploads/artifacts
  -> Elasticsearch if hybrid retrieval is enabled
  -> MLflow/Airflow if MLOps stack is enabled
```

## 26. MLOps Design

This project includes MLOps ideas around ingestion, clustering, autocomplete, monitoring, and artifact tracking.

### 26.1 DVC

File:

```text
python_doc_service/dvc.yaml
```

DVC stages include:

- cluster training
- autocomplete artifact tracking

Why DVC:

- track pipeline inputs and outputs
- make model/data processing reproducible
- store versioned artifacts

### 26.2 MLflow

MLflow can log:

- clustering metrics
- selected UMAP/HDBSCAN parameters
- noise ratio
- chunk counts
- autocomplete metrics
- model artifacts

Why MLflow:

- compare experiments
- debug model quality
- track what changed between runs

### 26.3 Airflow

Airflow is optional.

It can trigger:

- autocomplete training DAG
- HDBSCAN training DAG

Why Airflow:

- schedule retraining
- make pipelines observable
- separate long-running ML workflows from API requests

### 26.4 Monitoring

The Python service exposes:

```text
GET /monitoring/model-dashboard
```

Monitoring is useful for retrieval quality, model behavior, autocomplete quality, drift detection, and weak answer detection.

## 27. Security And Privacy

Current security strengths:

- passwords are hashed
- owner APIs use JWT protection
- tenant ID is checked on protected resources
- raw IP is hashed in chat sessions
- duplicate documents use hashes instead of name-only matching

Production improvements:

- move JWT from localStorage to secure HTTP-only cookies
- add rate limiting
- add request validation with a schema library
- scan uploaded files
- store uploads in S3/Azure Blob instead of local disk
- use a durable queue
- rotate secrets
- add audit logs for owner actions
- enforce stricter CORS
- encrypt sensitive data at rest
- add prompt-injection defenses and content filters

## 28. Scalability And Performance

Current optimizations:

- frontend autocomplete cache
- backend session cache
- backend suggestion cache
- backend conversation cache
- backend answer cache
- SHA-256 duplicate upload detection
- incremental rebuild mode
- hierarchical vector retrieval
- optional Elasticsearch hybrid retrieval
- social refresh scheduler

Scaling limitations:

| Area | Current state | Production improvement |
| --- | --- | --- |
| Upload queue | in-memory | Redis/BullMQ/SQS/RabbitMQ |
| FAISS indexes | local process memory + persisted files | managed vector DB or shared index service |
| Upload files | local disk | object storage |
| Backend cache | local memory | Redis |
| Social scraping | best-effort | official APIs where possible |
| Long ML jobs | API-triggered | dedicated worker pool |
| Multiple replicas | not fully coordinated | external queues and shared storage |

## 29. Design Tradeoffs

### 29.1 Why RAG Instead Of Fine-Tuning?

RAG was chosen because business documents change often.

Fine-tuning would be weaker here because:

- it is slower to update
- it may memorize stale information
- it does not naturally provide references
- it is expensive for every client/bot

RAG lets the system update knowledge by re-indexing documents.

### 29.2 Why Gemini Instead Of Only Extractive QA?

Extractive QA can return exact snippets, but support answers need readable explanations.

Gemini helps:

- summarize multiple chunks
- write natural responses
- combine document and memory context
- produce concise customer-friendly answers

The graph and prompts restrict Gemini to retrieved evidence.

### 29.3 Why Hybrid Retrieval?

Semantic search fails when exact words matter. BM25 fails when the user uses different wording. Hybrid retrieval combines both.

### 29.4 Why Human Escalation?

No AI support bot is perfect.

Escalation is important because:

- it prevents repeated bad experiences
- owners can correct knowledge gaps
- corrected answers become trusted memory
- it makes the product operationally useful

### 29.5 Why Multi-Tenant?

The product is a platform, not a single chatbot.

Multi-tenancy allows:

- many companies
- many bots per company
- isolated data
- reusable infrastructure
- SaaS-style business model

## 30. Known Gaps

These are honest limitations worth mentioning in interviews:

| Gap | Why it matters | Fix |
| --- | --- | --- |
| In-memory ingestion queue | jobs vanish on restart | use Redis/SQS/BullMQ |
| Public chat route protected in frontend | customer link may require login | make `/chat/:botId` public in React |
| Some Mongoose refs say `User` instead of `Client` | confusing model population | update refs to `Client` |
| Local file uploads | not safe across deployments | use S3/Azure Blob |
| Local FAISS process state | hard to scale horizontally | vector DB or shared index service |
| Token in localStorage | XSS risk | HTTP-only secure cookie |
| No strong rate limiting | abuse risk | add rate limiter |
| Limited automated tests | regressions possible | add unit/integration/e2e tests |
| Social scraping fragility | platforms can block scraping | use official APIs or cached ingestion |
| DVC params can hold runtime values | accidental dirty config | generate runtime params outside committed template |

Good interview answer:

This project is production-style but still has prototype-level infrastructure in a few places. The architecture already shows where durable queues, shared storage, stronger auth, and managed indexes should be added.

## 31. Testing Strategy

Recommended test layers:

| Layer | What to test |
| --- | --- |
| Frontend unit tests | route guards, chat rendering, upload states |
| Backend unit tests | auth, tenant checks, analytics calculations |
| Backend integration tests | MongoDB routes, document upload, feedback, retry |
| Python unit tests | chunking, clustering score, retrieval fusion |
| Python integration tests | `/process`, `/answer`, autocomplete |
| E2E tests | register, create bot, upload docs, chat, feedback, escalation |
| Load tests | concurrent chat sessions and ingestion jobs |

Most important tests to add first:

1. Tenant isolation: one client cannot access another client's bot.
2. Upload deduplication: same PDF is skipped.
3. Low-confidence answer: escalates instead of hallucinating.
4. Feedback retry: secondary retrieval excludes previous chunks.
5. Owner resolution: future similar questions use trusted answer.

## 32. Interview Questions And Answers

### What problem does Custom Care Bot solve?

It reduces repetitive customer support work by giving businesses a bot that answers from their own documents, website, social context, and previous support experiences. It also escalates weak answers so humans can correct the system.

### Is this just a PDF chatbot?

No. A PDF chatbot only retrieves from uploaded files. This project adds multi-tenancy, bot management, feedback, retry, escalation, owner memory, autocomplete training, analytics, website/social context, and MLOps tracking.

### What is the architecture?

React frontend, Express backend, MongoDB database, and FastAPI AI service. The backend handles product state and orchestration. The Python service handles document intelligence, retrieval, ML models, and LLM answering.

### Why use MongoDB?

The data is document-shaped: clients, bots, sessions, messages, experiences, and escalations. MongoDB is flexible for evolving schemas and works well with Mongoose in a Node backend.

### How is multi-tenancy handled?

The JWT contains `clientId`. Protected routes use that ID to query only the current client's bots, documents, analytics, and escalations.

### How are documents processed?

PDFs are uploaded to the backend, deduplicated by SHA-256 hash, then sent to the Python service. Python extracts text, detects headings, builds sections, removes boilerplate, creates chunks, embeds chunks, clusters them, and builds indexes.

### Why remove boilerplate?

Repeated low-value text can dominate retrieval. Removing it makes search return meaningful support content instead of footers, navigation, legal fragments, or repeated banners.

### Which embedding model is used?

`sentence-transformers/all-MiniLM-L6-v2`. It creates 384-dimensional embeddings and is fast enough for local/prototype deployment.

### Why use FAISS?

FAISS is fast for vector similarity search. It lets the service find chunks semantically related to the user's question.

### Why use UMAP and HDBSCAN?

UMAP reduces embedding dimensions while preserving neighborhoods. HDBSCAN clusters topics without requiring a fixed number of clusters and can mark outliers as noise.

### Why not K-Means?

K-Means requires choosing the number of clusters in advance and forces every chunk into a cluster. Support documents have unknown topic counts and noisy chunks, so HDBSCAN is a better fit.

### How does hybrid retrieval work?

Vector retrieval finds semantic matches. Elasticsearch BM25 finds exact keyword matches. The system combines both scores using configurable weights.

### What does LangGraph do?

LangGraph controls the answer workflow. It classifies intent, checks feedback state, searches memory, retrieves documents, analyzes confidence, optionally uses tools, escalates if needed, and finalizes the answer.

### Why use a graph instead of one chain?

A graph makes branching explicit. The system can choose different paths for retry, latest social updates, website questions, low-confidence answers, or owner-resolved memory.

### How does the system reduce hallucination?

It retrieves business context before generation, tells Gemini to use only provided context, checks retrieval confidence, returns "not enough verified information" when evidence is weak, and escalates repeated failures.

### What is an Experience?

An Experience is a saved question-answer event with retrieval metadata and feedback. It lets the system remember good answers, avoid bad ones, and use owner corrections.

### How does retry work?

If the user dislikes an answer and retry is allowed, the backend calls the Python answer endpoint with `retrieval_variant=secondary` and excludes chunks used in the weak answer. This forces a different retrieval attempt.

### When is escalation created?

Escalation is created when the system has weak evidence, repeated negative feedback, or no verified information. The owner can then provide the correct answer.

### What happens after the owner resolves an escalation?

The owner answer is stored as an Experience with high feedback score and `resolvedByOwner=true`. It is indexed in vector memory so similar future questions can use the trusted answer.

### How does autocomplete work?

The system records user questions per bot, combines them with generated starter questions, and trains a bot-specific model using SentencePiece, token embeddings, character encoding, and transformer-style decoder blocks.

### Why bot-specific autocomplete?

Different businesses have different question patterns. A healthcare bot, HR bot, and ecommerce bot should not suggest the same questions.

### What are DVC, MLflow, and Airflow used for?

DVC tracks pipeline stages and artifacts. MLflow logs metrics and experiments. Airflow can schedule or trigger longer ML workflows like clustering and autocomplete training.

### What is the biggest current production risk?

The biggest risks are in-memory queues/caches, local file/index storage, limited tests, and frontend public chat route protection. These are normal prototype tradeoffs and have clear production fixes.

### How would you scale this?

Use object storage for files, Redis or SQS for jobs, Redis for shared cache, managed Elasticsearch or vector DB for retrieval, separate worker services for ingestion, Kubernetes/App Service/ECS for scaling, and stronger observability.

### What makes the project unique?

It combines normal SaaS product features with RAG, graph-based routing, semantic memory, feedback-aware retries, human escalation, bot-specific autocomplete, and MLOps. It is closer to a real support platform than a simple chatbot demo.

## 33. Resume Pitch

Custom Care Bot is a production-style multi-tenant AI support platform built with React, Express, MongoDB, and FastAPI. It uses retrieval-augmented generation over uploaded PDFs, website content, and social context, with SentenceTransformer embeddings, FAISS, optional Elasticsearch hybrid retrieval, UMAP/HDBSCAN clustering, LangGraph orchestration, Gemini generation, feedback-aware experience memory, human escalation, bot-specific autocomplete, and MLOps tracking through DVC, MLflow, and Airflow.

## 34. End-To-End Demo Script

Use this flow in a demo or interview to explain the project naturally.

### Step 1: Register A Business

Say:

```text
First, a company registers. The backend stores the company as a Client and hashes the password using bcrypt. On login, the backend returns a JWT containing clientId, which is used for tenant isolation.
```

Technical points:

- frontend page: `Register.jsx`
- backend route: `POST /api/auth/register`
- model: `Client`
- password storage: bcrypt hash
- auth token: JWT

### Step 2: Create A Bot

Say:

```text
The owner creates a bot for a specific purpose, like customer support or HR onboarding. The bot is linked to the owner's tenantId, so another company cannot access it.
```

Technical points:

- frontend page: `CreateBot.jsx`
- backend route: `POST /api/bots`
- model: `Bot`
- tenant isolation: `tenantId = req.clientId`

### Step 3: Upload PDFs

Say:

```text
When the owner uploads PDFs, the backend first checks ownership, hashes files to skip duplicates, stores document metadata, and queues an ingestion job. The heavy ML processing is delegated to the Python service.
```

Technical points:

- frontend page: `BotDetail.jsx`
- backend route: `POST /api/documents/upload`
- upload library: multer
- deduplication: SHA-256 content hash
- job queue: in-memory queue
- Python endpoint: `POST /process`

### Step 4: Build Knowledge Index

Say:

```text
The Python service extracts PDF text, detects headings, removes boilerplate, chunks content, embeds chunks with all-MiniLM-L6-v2, clusters chunks with UMAP and HDBSCAN, and stores searchable indexes in FAISS and optionally Elasticsearch.
```

Technical points:

- extraction: PyMuPDF
- sentence processing: spaCy
- embeddings: SentenceTransformer
- clustering: UMAP + HDBSCAN
- vector index: FAISS
- lexical search: Elasticsearch BM25
- tracking: DVC and MLflow

### Step 5: Ask A Question

Say:

```text
When a customer asks a question, the backend saves the user message and sends the query, bot ID, client ID, and recent conversation to the Python answer service. The Python service uses LangGraph to classify intent, retrieve evidence, check memory, and generate a grounded answer.
```

Technical points:

- frontend component: `ChatWindow.jsx`
- backend route: `POST /api/chat/message`
- Python endpoint: `POST /answer`
- graph: LangGraph
- LLM: Gemini

### Step 6: Show Feedback Loop

Say:

```text
If the answer is useful, feedback increases the score for that experience. If the answer is bad, the score decreases. The user can retry, which triggers secondary retrieval and avoids the chunks that caused the poor answer. If the issue repeats, it becomes an escalation.
```

Technical points:

- model: `Experience`
- model: `Escalation`
- retry endpoint: `POST /api/chat/retry`
- feedback endpoint: `POST /api/chat/feedback`
- owner answer score: `OWNER_RESOLVED_SCORE = 5`

### Step 7: Owner Resolves Escalation

Say:

```text
The owner can resolve an escalation by writing the correct answer. That human answer is stored as trusted experience memory, so future similar questions can use it.
```

Technical points:

- backend route: `POST /api/escalations/resolve`
- trusted answer: `resolvedByOwner=true`
- memory indexing: `POST /experience/index`

## 35. Graphs In The Project

The word "graph" appears in two important ways.

### 35.1 LangGraph Control Graph

This is the decision graph for answering questions.

```text
Question
  -> classify intent
  -> check feedback history
  -> search experience memory
  -> retrieve documents/website/social context
  -> analyze confidence
  -> decide answer, retry, tool lookup, or human escalation
  -> generate final response
```

Why this graph matters:

- it makes decision-making explicit
- it supports branches instead of one fixed chain
- it separates retrieval, analysis, memory, tools, and generation
- it is easier to debug because the frontend can show a trace

### 35.2 Analytics Graphs

The dashboard can turn backend analytics into visual graphs such as:

- sessions over time
- user messages vs assistant messages
- negative feedback count
- retry rate
- escalation rate
- unresolved questions
- top negative hotspots
- no-document answer rate

Even if the UI currently shows simple dashboard cards/lists, these are the natural graphs to add because the backend already computes the raw metrics.

### 35.3 Knowledge Graph vs Current Design

This project does not use a formal entity knowledge graph like Neo4j.

Instead, it uses:

- semantic vector indexes
- cluster hierarchy
- LangGraph workflow graph
- experience memory grouping

If asked why not a knowledge graph:

```text
A formal knowledge graph is useful when relationships between entities are the main data model. Here, the main problem is retrieving policy/document passages and answering natural-language questions. Vector retrieval plus graph-based workflow routing is simpler and more directly useful for this product.
```

## 36. Model Inventory

The project uses two meanings of "model": database models and machine learning models.

### 36.1 Database Models

| Model | Location | Purpose |
| --- | --- | --- |
| `Client` | `server/models/Client.js` | Company account |
| `Bot` | `server/models/Bot.js` | One support bot owned by a client |
| `Document` | `server/models/Document.js` | Uploaded PDF metadata |
| `ChatSession` | `server/models/ChatSession.js` | One conversation |
| `ChatMessage` | `server/models/ChatMessage.js` | User/assistant messages |
| `Experience` | `server/models/Experience.js` | Answer memory with feedback |
| `Escalation` | `server/models/Escalation.js` | Human follow-up queue |

### 36.2 ML And AI Models

| Model/system | Purpose | Why used |
| --- | --- | --- |
| `all-MiniLM-L6-v2` | Converts text into embeddings | Fast semantic search |
| UMAP | Reduces embedding dimensions | Better clustering |
| HDBSCAN | Clusters document chunks | Unknown number of topics |
| TF-IDF | Finds key terms and boilerplate | Lightweight text statistics |
| FAISS | Searches vectors | Fast retrieval |
| Elasticsearch BM25 | Lexical retrieval | Exact keyword matching |
| Gemini | Generates final answers | Natural language response |
| LangGraph | Routes answer workflow | Branching decision control |
| SentencePiece | Tokenizes autocomplete text | Handles subwords |
| Keras autocomplete model | Predicts query continuations | Bot-specific suggestions |

## 37. Important Metrics And KPIs

In interviews, connect technical metrics to business value.

| Metric | Meaning | Business value |
| --- | --- | --- |
| Containment rate | Percent of questions answered without human help | Shows support workload reduction |
| Escalation rate | Percent of chats needing owner help | Shows knowledge gaps |
| Retry success rate | Percent of retries that produce better answers | Shows retrieval improvement |
| Negative feedback rate | Percent of disliked answers | Proxy for poor customer experience |
| No-doc rate | Questions where no evidence is found | Shows missing documentation |
| Average confidence | Retrieval/answer confidence signal | Helps monitor answer quality |
| Top unresolved questions | Frequent questions without good answers | Tells owner what to document |
| Autocomplete acceptance | Suggestions clicked or used | Measures autocomplete usefulness |
| Ingestion time | Time from upload to searchable bot | Measures operational speed |
| Answer latency | Time from user question to response | Affects customer experience |

Good interview answer:

```text
I would not judge this project only by model accuracy. For a support product, I would track containment rate, escalation rate, negative feedback rate, retry success, answer latency, and no-document rate because these directly show whether the bot is reducing support work safely.
```

## 38. Edge Cases And How The System Handles Them

| Edge case | Current behavior |
| --- | --- |
| Same PDF uploaded twice | Backend uses SHA-256 hash and skips duplicate content |
| Bot has no documents | Retrieval confidence becomes weak and answer may route to no-doc/human flow |
| Python service is down | Backend RAG call fails; production should return friendly fallback and alert |
| Gemini key missing | Python has fallback behavior, but production should fail clearly and monitor it |
| Elasticsearch is down | System can continue with FAISS semantic retrieval |
| User dislikes answer | Feedback decreases score and retry may become available |
| User dislikes answer repeatedly | Escalation is created |
| Owner resolves issue | Trusted answer is stored and indexed |
| Website crawl fails | PDF/document retrieval can still work |
| Social platform blocks scraping | Cached social context or no social result is used |
| Backend restarts during ingestion | In-memory job is lost; this is a known production gap |
| Multiple backend replicas | Local queue/cache can diverge; use shared queue/cache in production |
| Customer asks unrelated question | System should say it lacks verified information instead of hallucinating |

## 39. Prompting Strategy

The final answer prompt is designed for support safety.

Core rules:

- answer only from provided docs/tools/memory
- do not invent policy
- if evidence is missing, say verified information is not available
- keep the response concise and factual
- prefer owner-resolved memory when available
- prefer social/tool context for latest-update questions

Why this matters:

Generic LLM prompts often produce confident answers even with weak evidence. This project tries to make the LLM a writer over retrieved evidence, not an independent source of truth.

## 40. Backend-To-Python Contract

The backend sends the Python service enough context to answer safely.

Typical answer request:

```json
{
  "query": "What is your refund policy?",
  "bot_id": "bot_id",
  "client_id": "client_id",
  "top_k": 3,
  "retrieval_variant": "primary",
  "exclude_chunk_refs": [],
  "conversation": [],
  "website_url": "https://example.com",
  "social_links": {
    "facebook": "",
    "instagram": ""
  }
}
```

Important contract rules:

- `client_id` and `bot_id` isolate indexes
- `conversation` gives short-term context
- `retrieval_variant` controls primary vs secondary retrieval
- `exclude_chunk_refs` prevents retry from using the same weak evidence
- `website_url` and `social_links` help source-aware retrieval

## 41. Delete Flow

Deleting a bot is not only a MongoDB delete.

Current flow:

```text
owner deletes bot
  -> backend verifies ownership
  -> backend calls Python /bot/delete
  -> Python deletes indexes, chunks, website data, experience vectors, ES docs
  -> backend deletes uploaded files
  -> backend deletes documents, escalations, experiences, messages, sessions
  -> backend deletes bot
```

Why call Python first:

If the AI indexes are not deleted, stale private bot data may remain searchable. The backend waits for Python cleanup before deleting the product records.

## 42. How To Explain The Thought Process

Use this structure when explaining design decisions:

```text
1. Start with the business pain.
2. Explain why a normal FAQ bot is not enough.
3. Introduce multi-tenancy because many companies need isolated bots.
4. Introduce RAG because knowledge changes frequently.
5. Introduce feedback memory because answers must improve.
6. Introduce human escalation because AI will never be perfect.
7. Introduce MLOps because ingestion and models need tracking.
8. Finish with production gaps and how you would fix them.
```

Example answer:

```text
I designed it as a support platform, not just a chatbot. The business need is to reduce repeated support questions while keeping answers grounded. That is why I used RAG over company documents, tenant isolation for multiple businesses, feedback memory to learn from previous answers, and escalation so weak answers become human-corrected knowledge. The Python service handles ML-heavy work, while the Node backend owns product state.
```

## 43. What I Would Improve Next

Highest-value improvements:

1. Make `/chat/:botId` public in the frontend for real customer access.
2. Replace in-memory ingestion queue with Redis/BullMQ or SQS.
3. Move uploaded files and generated artifacts to S3 or Azure Blob.
4. Add rate limiting and request validation.
5. Add automated integration tests for tenant isolation and feedback flow.
6. Replace local FAISS state with a managed vector DB or shared retrieval service for multi-replica scaling.
7. Add a real embeddable widget script for customer websites.
8. Add RBAC so support agents and admins have different permissions.
9. Add an evaluation dataset with expected answers per bot.
10. Add prompt-injection tests for uploaded and website content.

Medium-term improvements:

- billing and usage limits
- organization/team management
- Slack/WhatsApp/email handoff
- official social APIs
- scheduled website recrawling
- versioned document rollbacks
- per-bot answer style settings
- multilingual support
- audit logs
- dashboard charts

## 44. Short Interview Answers

### Explain the backend in 20 seconds.

The backend is an Express API that handles auth, tenant isolation, bot management, document upload orchestration, chat persistence, feedback, retry, escalation, and analytics. It stores product data in MongoDB and delegates ML/RAG work to the Python service.

### Explain the ML pipeline in 20 seconds.

The Python service extracts PDF and website text, removes boilerplate, chunks content, embeds chunks with `all-MiniLM-L6-v2`, clusters document chunks with UMAP/HDBSCAN, builds FAISS and optional Elasticsearch indexes, then uses LangGraph and Gemini to answer questions from retrieved context.

### Explain the feedback loop in 20 seconds.

Every answer becomes an Experience with retrieval metadata and feedback score. Positive feedback increases trust, negative feedback lowers it, retry uses different chunks, and repeated failures create escalations. Owner-resolved answers become trusted memory for future similar questions.

### Explain why this is production-style.

It includes authentication, multi-tenancy, upload deduplication, async ingestion, retrieval, fallback logic, feedback, escalation, analytics, deployment docs, DVC, MLflow, and optional Airflow. It still has known prototype gaps like in-memory queues and local storage, but the architecture shows a clear production path.

### Explain the biggest technical challenge.

The hardest part is reliable answer quality. The system must retrieve the right evidence, avoid hallucination, detect weak answers, learn from feedback, and let humans correct knowledge gaps. That is why the project combines RAG, graph routing, experience memory, retry, and escalation.

## 45. Glossary

| Term | Simple meaning |
| --- | --- |
| RAG | Retrieve relevant knowledge first, then generate an answer |
| Embedding | Numeric representation of text meaning |
| Vector search | Search by semantic similarity |
| FAISS | Library for fast vector search |
| BM25 | Keyword-based search algorithm |
| Hybrid retrieval | Combining vector search and keyword search |
| Chunk | Small piece of document text |
| Boilerplate | Repeated low-value text like footers or navigation |
| UMAP | Reduces embedding dimensions |
| HDBSCAN | Clusters data without fixed number of clusters |
| LangGraph | Graph-based LLM workflow framework |
| Experience | Saved answer event with feedback and retrieval metadata |
| Escalation | Question needing human correction |
| DVC | Data/model pipeline versioning |
| MLflow | Experiment and metric tracking |
| Airflow | Workflow scheduler/orchestrator |
| Tenant | One company/client using the platform |

## 46. Simple Final Summary

This project answers customer questions using a company's own knowledge. It does not trust the LLM alone. It retrieves evidence, checks memory, generates a grounded answer, learns from feedback, escalates weak answers, and helps the owner improve the bot over time.
