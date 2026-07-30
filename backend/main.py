from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Header
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from services.document_service import document_service
from services.rag_service import rag_service
from worker import process_document_task
from api.auth import router as auth_router

app = FastAPI(
    title="DocuAI Enterprise RAG API",
    description="High-performance backend for semantic document retrieval and ingestion.",
    version="1.0.0"
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

@app.post("/api/v1/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    x_session_id: str = Header(default="default_session")
):
    """
    Handles PDF uploads, saves to MinIO object storage, 
    and enqueues background processing scoped to session_id.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        doc_id, filename, file_uri = document_service.upload_document(file, session_id=x_session_id)
        
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
def list_documents(x_session_id: str = Header(default="default_session")):
    return document_service.list_documents(session_id=x_session_id)

@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str, x_session_id: str = Header(default="default_session")):
    success = document_service.delete_document(doc_id, session_id=x_session_id)
    if success:
        return {"message": "Document deleted"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/api/v1/documents/{doc_id}/url")
def get_document_url(doc_id: str, x_session_id: str = Header(default="default_session")):
    url = document_service.get_presigned_url(doc_id, session_id=x_session_id)
    if url:
        return {"url": url}
    raise HTTPException(status_code=404, detail="URL generation failed")

@app.get("/api/v1/documents/{doc_id}/download")
def download_document(doc_id: str, x_session_id: str = Header(default="default_session")):
    docs = document_service.list_documents(session_id=x_session_id)
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if doc and os.path.exists(doc["key"]):
        return FileResponse(path=doc["key"], filename=doc["name"], media_type="application/pdf")
    raise HTTPException(status_code=404, detail="File not found")

class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

@app.post("/api/v1/chat")
async def chat_with_documents(
    request: ChatRequest,
    x_session_id: str = Header(default="default_session")
):
    """
    Streaming endpoint for querying the RAG system scoped to session_id.
    """
    async def generate():
        async for chunk in rag_service.stream_response(request.query, request.document_ids, session_id=x_session_id):
            yield chunk
            
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
