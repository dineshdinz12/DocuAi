# DocuAI - Serverless Enterprise Multi-Document RAG Platform

DocuAI is a high-performance, serverless Retrieval-Augmented Generation (RAG) platform designed for multi-document intelligence and automated question-answering over PDF files. 

Version 2 introduces a zero-cost cloud architecture that eliminates heavy local container dependencies (such as local Ollama, Redis, or Celery) in favor of high-throughput serverless APIs (**Groq**, **Hugging Face**, **Qdrant Cloud**) and **Multi-Tenant User Session Isolation**.

---

## Technical Overview

- **Serverless LLM Inference**: Integrated with the **Groq API** (`llama-3.1-8b-instant`) for sub-second, low-latency streaming responses.
- **Cloud Vector Embeddings**: Utilizes the **Hugging Face Inference API** (`sentence-transformers/all-MiniLM-L6-v2`) to produce 384-dimensional dense vector embeddings without local GPU/CPU overhead.
- **Managed Vector Storage**: Cloud-native similarity search and metadata filtering powered by **Qdrant Cloud**.
- **Multi-Tenant Session Isolation**: Client-scoped session management (`x-session-id`) ensures strict user-level document privacy and isolated vector retrieval.
- **Asynchronous Processing Pipeline**: Native **FastAPI BackgroundTasks** execution replaces legacy Celery worker architectures for serverless environment compatibility.
- **Enterprise Web Interface**: Next.js 15 App Router interface built with TypeScript, Tailwind CSS, custom branding, and interactive context selection.

---

## System Architecture

```
                               ┌─────────────────────────┐
                               │     Next.js 15 UI       │
                               │   (Vercel / Local)      │
                               └────────────┬────────────┘
                                            │  x-session-id
                                            ▼
                               ┌─────────────────────────┐
                               │     FastAPI Backend     │
                               │   (Render / Local)      │
                               └──────┬───────────┬──────┘
                                      │           │
            ┌─────────────────────────┴─┐       ┌─┴────────────────────────┐
            │   Background Processing   │       │     Streaming Chat RAG   │
            └─────────────┬─────────────┘       └──────────┬───────────────┘
                          │                                │
            ┌─────────────┴─────────────┐       ┌──────────┴───────────────┐
            │ HuggingFace Embeddings API │       │      Groq Llama 3.1      │
            └─────────────┬─────────────┘       └──────────────────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   Qdrant Cloud Vector DB  │
            │ (Scoped by x-session-id)  │
            └───────────────────────────┘
```

---

## Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router, TypeScript)
- **Styling**: Tailwind CSS
- **Component Primitives**: Radix UI / Lucide React
- **Rendering**: React Markdown with GitHub Flavored Markdown (GFM)

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **LLM Engine**: Groq API (`llama-3.1-8b-instant`)
- **Embedding Provider**: Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: Qdrant Cloud
- **Object Storage**: MinIO / Amazon S3
- **Orchestration**: LangChain Framework

---

## Multi-Tenant Security & Isolation

To ensure data privacy in shared environment deployments:
1. The frontend automatically assigns a unique, cryptographically generated session ID in browser local storage (`docuai_session_id`).
2. HTTP requests forward the session identifier via the custom `x-session-id` header.
3. Uploaded documents are stored under user-specific bucket prefixes (`users/{session_id}/`).
4. Vector embeddings uploaded to Qdrant Cloud are indexed with `session_id` payload metadata.
5. Similarity search queries apply a strict mandatory filter (`FieldCondition`) enforcing `session_id` matching, preventing cross-tenant data exposure.

---

## Local Development Setup

### 1. Repository Setup

```bash
git clone https://github.com/dineshdinz12/DocuAi.git
cd DocuAi
git checkout version_2
```

### 2. Environment Configuration

Create `backend/.env` with the required configuration parameters:

```env
# Cloud API Credentials
GROQ_API_KEY=gsk_your_groq_api_key
HF_TOKEN=hf_your_huggingface_token
MAIN_LLM_MODEL=llama-3.1-8b-instant

# Vector Database (Qdrant Cloud)
QDRANT_HOST=https://your-cluster-id.eu-west-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key

# Object Storage Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents
```

### 3. Object Storage Initialization

Start a local MinIO instance via Docker:

```bash
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

### 4. Backend Execution

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

### 5. Frontend Execution

```bash
cd frontend
npm install
npm run dev
```

Access the application at `http://localhost:3000`.

---

## Production & Serverless Deployment

### Frontend (Vercel)
1. Import the repository into Vercel.
2. Select `frontend` as the Root Directory.
3. Configure the build command as `npm run build` and deploy.

### Backend (Render / Railway)
1. Deploy the `backend/` directory as a Web Service.
2. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Set the production environment variables (`GROQ_API_KEY`, `HF_TOKEN`, `QDRANT_HOST`, `QDRANT_API_KEY`).

---

## License

This project is licensed under the MIT License.