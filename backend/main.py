import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json

from services.document_service import document_service
from services.rag_service import rag_service
from services.supabase_service import supabase_service
from worker import process_document_task
from api.auth import router as auth_router

app = FastAPI(
    title="DocuAI Enterprise RAG API",
    description="High-performance backend for semantic document retrieval and ingestion.",
    version="2.0.0"
)

app.include_router(auth_router)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DocuAI API"}

# ─── Documents ────────────────────────────────────────────────────────────────

@app.post("/api/v1/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    x_session_id: str = Header(default="default_session"),
    x_user_id: Optional[str] = Header(default=None)
):
    """
    Handles PDF uploads, saves to MinIO/local storage, 
    persists metadata to Supabase, and enqueues background processing.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        doc_id, filename, file_uri = document_service.upload_document(file, session_id=x_session_id)
        
        # Persist document metadata to Supabase
        file_size = 0
        try:
            if os.path.exists(file_uri):
                file_size = os.path.getsize(file_uri)
        except Exception:
            pass
        
        supabase_service.save_document(
            doc_id=doc_id,
            name=filename,
            storage_key=file_uri,
            size=file_size,
            session_id=x_session_id,
            user_id=x_user_id
        )
        
        # Enqueue background task with session_id
        background_tasks.add_task(process_document_task, doc_id, file_uri, session_id=x_session_id)
        
        return {
            "message": "Document uploaded and processing started.",
            "document_id": doc_id,
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.get("/api/v1/documents")
def list_documents(
    x_session_id: str = Header(default="default_session"),
    x_user_id: Optional[str] = Header(default=None)
):
    # Try Supabase first (logged-in users get cross-device sync)
    sb_docs = supabase_service.list_documents(session_id=x_session_id, user_id=x_user_id)
    if sb_docs is not None and len(sb_docs) > 0:
        return sb_docs
    # Fall back to local disk/MinIO listing
    return document_service.list_documents(session_id=x_session_id)

@app.delete("/api/v1/documents/{doc_id}")
def delete_document(
    doc_id: str, 
    x_session_id: str = Header(default="default_session")
):
    success = document_service.delete_document(doc_id, session_id=x_session_id)
    if success:
        rag_service.delete_document_vectors(doc_id)
        supabase_service.delete_document(doc_id=doc_id, session_id=x_session_id)
        return {"message": "Document deleted"}
    # Also try Supabase-only delete (if file already removed but DB record remains)
    supabase_service.delete_document(doc_id=doc_id, session_id=x_session_id)
    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/api/v1/documents/{doc_id}/url")
def get_document_url(doc_id: str, x_session_id: str = Header(default="default_session")):
    url = document_service.get_presigned_url(doc_id, session_id=x_session_id)
    if url:
        return {"url": url}
    raise HTTPException(status_code=404, detail="URL generation failed")

@app.get("/api/v1/documents/{doc_id}/download")
def download_document(
    doc_id: str, 
    session_id: Optional[str] = None, 
    x_session_id: str = Header(default="default_session")
):
    actual_session_id = session_id or x_session_id
    docs = document_service.list_documents(session_id=actual_session_id)
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if doc and os.path.exists(doc["key"]):
        # Content-Disposition: attachment -> forces download
        return FileResponse(
            path=doc["key"],
            filename=doc["name"],
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{doc["name"]}"'}
        )
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/v1/documents/{doc_id}/view")
def view_document(
    doc_id: str,
    session_id: Optional[str] = None,
    x_session_id: str = Header(default="default_session")
):
    """Serve PDF inline so the browser opens it in a new tab instead of downloading."""
    actual_session_id = session_id or x_session_id
    docs = document_service.list_documents(session_id=actual_session_id)
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if doc and os.path.exists(doc["key"]):
        # Content-Disposition: inline -> browser renders the PDF
        return FileResponse(
            path=doc["key"],
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{doc["name"]}"'}
        )
    raise HTTPException(status_code=404, detail="File not found")

# ─── Chat ─────────────────────────────────────────────────────────────────────

class MessageModel(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    history: Optional[List[MessageModel]] = None
    chat_id: Optional[str] = None       # Phase B: Supabase session ID
    user_message_id: Optional[str] = None   # frontend-generated ID for persistence
    bot_message_id: Optional[str] = None

@app.post("/api/v1/chat")
async def chat_with_documents(
    request: ChatRequest,
    x_session_id: str = Header(default="default_session"),
    x_user_id: Optional[str] = Header(default=None)
):
    """
    Streaming SSE endpoint for RAG queries. Saves messages to Supabase.
    """
    full_response = []
    response_sources = []

    async def generate():
        nonlocal response_sources
        async for chunk in rag_service.stream_response(
            request.query, 
            request.document_ids, 
            session_id=x_session_id,
            history=request.history
        ):
            # Parse to capture sources for persistence
            try:
                data = json.loads(chunk.removeprefix("data: ").strip())
                if "text" in data:
                    full_response.append(data["text"])
                elif "citations" in data:
                    response_sources = data["citations"]
            except Exception:
                pass
            yield chunk
        
        # Phase B: persist both messages to Supabase after stream completes
        if request.chat_id:
            # Save user message
            if request.user_message_id:
                supabase_service.save_message(
                    msg_id=request.user_message_id,
                    chat_id=request.chat_id,
                    role="user",
                    content=request.query,
                    sources=[]
                )
            # Save assistant message  
            if request.bot_message_id:
                supabase_service.save_message(
                    msg_id=request.bot_message_id,
                    chat_id=request.chat_id,
                    role="assistant",
                    content="".join(full_response),
                    sources=response_sources
                )
            
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ─── Chat Sessions (Phase B) ──────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    chat_id: str
    title: str

@app.post("/api/v1/sessions")
def create_session(
    req: CreateSessionRequest,
    x_session_id: str = Header(default="default_session"),
    x_user_id: Optional[str] = Header(default=None)
):
    """Create or update a named chat session in Supabase."""
    supabase_service.create_or_update_session(
        session_id=x_session_id,
        chat_id=req.chat_id,
        title=req.title,
        user_id=x_user_id
    )
    return {"status": "ok", "chat_id": req.chat_id}

@app.get("/api/v1/sessions")
def list_sessions(
    x_session_id: str = Header(default="default_session"),
    x_user_id: Optional[str] = Header(default=None)
):
    """Return all chat sessions with their messages."""
    return supabase_service.list_sessions(session_id=x_session_id, user_id=x_user_id)

@app.get("/api/v1/sessions/{chat_id}/messages")
def get_session_messages(chat_id: str):
    """Return all messages for a specific chat session."""
    return supabase_service.get_messages(chat_id=chat_id)

@app.delete("/api/v1/sessions/{chat_id}")
def delete_session(chat_id: str):
    supabase_service.delete_session(chat_id=chat_id)
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
