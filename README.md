# DocuAI

> Serverless Enterprise Multi-Document RAG Platform

DocuAI is a high-performance, serverless Retrieval-Augmented Generation (RAG) platform engineered for multi-document intelligence and automated semantic question-answering over PDF files.

---

## Architectural Highlights

- **Unified Single-Port Ingress**: Exposes a single public interface on port `3000` for both the Next.js UI and FastAPI endpoints via internal reverse-proxy rewrites, eliminating cross-origin resource sharing (CORS) security overhead.
- **One-Command Process Orchestration**: Integrated task runner (`make dev` or `npm run dev:all`) that launches Uvicorn FastAPI and Next.js concurrently.
- **Serverless LLM Inference**: Low-latency streaming responses powered by the Groq Inference Engine (`llama-3.1-8b-instant`).
- **Cloud Vector Storage**: 384-dimensional dense vector embeddings generated via Hugging Face Inference APIs and indexed in Qdrant Cloud.
- **Multi-Tenant Session Isolation**: Client-scoped session management (`x-session-id`) enforcing strict user-level document privacy.
- **Asynchronous Processing Pipeline**: Native FastAPI `BackgroundTasks` execution replacing heavy worker queues.

---

## System Architecture

```
                       SINGLE PUBLIC PORT (:3000)
                                    │
                     ┌──────────────┴──────────────┐
                     │   Next.js 15 App Router     │
                     │  (Client & Ingress Proxy)   │
                     └──────────────┬──────────────┘
            ┌───────────────────────┴───────────────────────┐
            │  x-session-id Header                         │ /api/* Rewrites
            ▼                                               ▼
┌─────────────────────────┐                     ┌─────────────────────────┐
│     Next.js Frontend    │                     │     FastAPI Backend     │
│   (React 19 / Tailwind) │                     │     (Internal :8000)    │
└─────────────────────────┘                     └────────────┬────────────┘
                                                             │
                                   ┌─────────────────────────┴─────────────────────────┐
                                   │                                                   │
                     ┌─────────────┴─────────────┐                       ┌─────────────┴─────────────┐
                     │ HuggingFace Embeddings    │                       │ Groq LLM Inference        │
                     └─────────────┬─────────────┘                       └───────────────────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │ Qdrant Cloud Vector DB    │
                     │ (Scoped by x-session-id)  │
                     └───────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Description |
| :--- | :--- | :--- |
| **Frontend Framework** | Next.js 15 (App Router) | React 19, TypeScript |
| **Styling & UI** | Tailwind CSS | Lucide React primitives, GFM markdown |
| **Backend API** | FastAPI (Python 3.11+) | Asynchronous execution, BackgroundTasks |
| **LLM Inference** | Groq API | `llama-3.1-8b-instant` streaming model |
| **Embeddings** | Hugging Face Inference API | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Database** | Qdrant Cloud | Metadata-filtered similarity search |
| **Object Storage** | MinIO / AWS S3 | Client-scoped bucket prefixes |

---

## Quickstart

### 1. Environment Configuration

Create `backend/.env` with the following configuration parameters:

```env
# Cloud API Credentials
GROQ_API_KEY=gsk_your_groq_api_key
HF_TOKEN=hf_your_huggingface_token
MAIN_LLM_MODEL=llama-3.1-8b-instant

# Vector Database (Qdrant Cloud)
QDRANT_HOST=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key

# Object Storage Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents
```

### 2. Local Development

Run the entire application stack using a single command:

```bash
# Option A: Root Makefile
make dev

# Option B: From frontend directory
cd frontend
npm install
npm run dev:all
```

Access the application at `http://localhost:3000`.

---

## Containerized Deployment

To deploy using Docker Compose exposing only port 3000:

```bash
# Start container environment
make docker-dev

# Stop container environment
make docker-down
```

---

## Production Cloud Deployment

### Frontend (Vercel)

1. Import the repository into Vercel.
2. In **Project Settings > Build and Deployment**:
   - Set **Root Directory** to `frontend`.
3. In **Environment Variables**:
   - Set `BACKEND_INTERNAL_URL` = `https://your-backend-service.onrender.com`
4. Deploy. Next.js handles `/api/*` rewrites server-side.

### Backend (Render / Railway)

1. Deploy `backend/` as a Python Web Service.
2. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Configure environment variables: `GROQ_API_KEY`, `HF_TOKEN`, `QDRANT_HOST`, `QDRANT_API_KEY`.

---

## Security and Multi-Tenancy

Data isolation is strictly enforced at every layer:

1. **Session Assignment**: The client initializes a unique `docuai_session_id` in browser storage.
2. **Request Scoping**: Requests forward the session identifier in the custom `x-session-id` HTTP header.
3. **Storage Prefixes**: Files are saved under tenant prefixes (`users/{session_id}/`).
4. **Vector Filters**: Qdrant queries execute mandatory metadata filtering on `session_id`, ensuring absolute cross-tenant isolation.

---

## License

Distributed under the MIT License.