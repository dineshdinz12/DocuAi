# DocuAI - Enterprise Multi-Document RAG Platform

DocuAI is a high-performance, serverless Retrieval-Augmented Generation (RAG) platform built for multi-document intelligence and automated semantic question-answering over PDF files.

Version 2 introduces a **Unified Single-Port Architecture** that eliminates CORS complexity, incorporates **Multi-Tenant User Session Isolation**, and leverages zero-cost serverless APIs (**Groq**, **Hugging Face**, **Qdrant Cloud**).

---

## 🚀 Key Architectural Features

- **Unified Single-Port Ingress**: Exposes a single public port (`:3000`) for both Frontend UI and Backend APIs via Next.js internal reverse proxy rewrites, eliminating cross-origin resource sharing (CORS) security overhead.
- **One-Command Process Orchestration**: Integrated process launcher (`make dev` or `npm run dev:all`) that boots both FastAPI and Next.js concurrently.
- **Serverless LLM Inference**: Sub-second streaming responses powered by **Groq API** (`llama-3.1-8b-instant`).
- **Cloud Vector Storage**: 384-dimensional dense vector embeddings generated via **Hugging Face Inference API** and indexed in **Qdrant Cloud**.
- **Multi-Tenant Session Isolation**: Client-scoped session management (`x-session-id`) enforcing strict user-level document privacy.
- **Asynchronous Processing Pipeline**: Native **FastAPI BackgroundTasks** execution replacing heavy worker queues.

---

## 🏗️ System Architecture

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

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router, TypeScript, React 19)
- **Styling**: Tailwind CSS
- **Primitives**: Lucide React Icons
- **Markdown**: React Markdown with GitHub Flavored Markdown (GFM)

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **LLM Engine**: Groq API (`llama-3.1-8b-instant`)
- **Embeddings**: Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: Qdrant Cloud
- **Object Storage**: MinIO / S3 compatible storage
- **Orchestration**: LangChain Framework

---

## ⚡ Quickstart (Local Development)

### 1. Environment Setup

Create `backend/.env`:

```env
# Cloud API Credentials
GROQ_API_KEY=gsk_your_groq_api_key
HF_TOKEN=hf_your_huggingface_token
MAIN_LLM_MODEL=llama-3.1-8b-instant

# Vector Database (Qdrant Cloud)
QDRANT_HOST=https://your-cluster-id.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key

# Object Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=documents
```

### 2. One-Command Developer Launch

Run the entire application stack using a single command:

```bash
# Option A: Using Makefile (Root)
make dev

# Option B: From frontend directory
cd frontend
npm install
npm run dev:all
```

Access the unified platform at **`http://localhost:3000`**.

---

## 🐳 Docker Deployment (Containerized)

To launch the full stack in Docker exposing **only port 3000**:

```bash
# Boot single-port container environment
make docker-dev

# Tear down containers
make docker-down
```

---

## 🌐 Production Cloud Deployment

### 1. Frontend Deployment (Vercel)
1. Import repository into Vercel.
2. In **Project Settings $\rightarrow$ Build & Deployment Settings**:
   - Set **Root Directory** to `frontend`.
3. In **Environment Variables**:
   - Set `BACKEND_INTERNAL_URL` = `https://your-backend-api.onrender.com`
4. Deploy! Next.js will automatically proxy `/api/*` traffic server-side.

### 2. Backend Deployment (Render / Railway)
1. Deploy `backend/` as a Web Service.
2. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables: `GROQ_API_KEY`, `HF_TOKEN`, `QDRANT_HOST`, `QDRANT_API_KEY`.

---

## 📄 License

This project is licensed under the MIT License.