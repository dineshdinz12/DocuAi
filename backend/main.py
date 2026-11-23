import json
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field

from core.config import settings
from core.security import CurrentUser, require_user
from services.document_service import document_service
from services.rag_service import rag_service
from services.supabase_service import supabase_service
from worker import process_document_task

app = FastAPI(title="DocuAI API", version="3.0.0", docs_url="/docs" if settings.APP_ENV != "production" else None)
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DocuAI API"}


@app.get("/ready")
def readiness_check():
    checks = {"supabase": supabase_service.is_configured(), "rag": rag_service.is_ready()}
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", "checks": checks}


def _owned_document(doc_id: str, user_id: str) -> dict:
    document = supabase_service.get_document(doc_id, user_id)
    if not document:
        document = next((item for item in document_service.list_documents(user_id) if item["id"] == doc_id), None)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.post("/api/v1/documents/upload", status_code=202)
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), user: CurrentUser = Depends(require_user)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".pdf" or file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size <= 0 or size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF must be between 1 byte and {settings.MAX_UPLOAD_MB} MB")
    if file.file.read(5) != b"%PDF-":
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")
    file.file.seek(0)
    try:
        doc_id, safe_name, storage_key = document_service.upload_document(file, session_id=user.id)
        if not supabase_service.save_document(doc_id, safe_name, storage_key, size, user.id, user.id):
            document_service.delete_document(doc_id, user.id)
            raise HTTPException(status_code=503, detail="Document metadata could not be saved")
        background_tasks.add_task(process_document_task, doc_id, storage_key, user.id, safe_name)
        return {"message": "Document accepted for processing", "document_id": doc_id, "filename": safe_name, "status": "processing"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document upload failed") from exc


@app.get("/api/v1/documents")
def list_documents(user: CurrentUser = Depends(require_user)):
    documents = supabase_service.list_documents(user.id, user.id)
    return documents if documents is not None else document_service.list_documents(user.id)


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str, user: CurrentUser = Depends(require_user)):
    document = _owned_document(doc_id, user.id)
    document_service.delete_document(doc_id, user.id)
    rag_service.delete_document_vectors(doc_id, user.id)
    if not supabase_service.delete_document(doc_id, user.id):
        raise HTTPException(status_code=503, detail="Document metadata deletion failed")
    return {"message": "Document deleted", "name": document.get("name")}


@app.get("/api/v1/documents/{doc_id}/content")
def document_content(doc_id: str, download: bool = False, user: CurrentUser = Depends(require_user)):
    document = _owned_document(doc_id, user.id)
    storage_key = document.get("storage_key") or document.get("key")
    if not storage_key:
        raise HTTPException(status_code=404, detail="Stored file not found")
    if document_service.use_s3:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        handle.close()
        try:
            path = document_service.download_to_path(storage_key, handle.name)
        except Exception as exc:
            os.unlink(handle.name)
            raise HTTPException(status_code=502, detail="Stored file could not be retrieved") from exc
    else:
        path = storage_key
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Stored file not found")
    disposition = "attachment" if download else "inline"
    cleanup = BackgroundTask(os.unlink, path) if document_service.use_s3 else None
    return FileResponse(path, media_type="application/pdf", filename=document["name"], headers={"Content-Disposition": f'{disposition}; filename="{document["name"]}"'}, background=cleanup)


class MessageModel(BaseModel):
    role: str
    content: str = Field(max_length=100_000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    document_ids: List[str] = Field(default_factory=list)
    history: List[MessageModel] = Field(default_factory=list)
    chat_id: str = Field(min_length=1, max_length=100)
    user_message_id: str = Field(min_length=1, max_length=100)
    bot_message_id: str = Field(min_length=1, max_length=100)


@app.post("/api/v1/chat")
async def chat(request: ChatRequest, user: CurrentUser = Depends(require_user)):
    if request.document_ids and not supabase_service.user_owns_documents(request.document_ids, user.id):
        raise HTTPException(status_code=403, detail="One or more documents are unavailable")
    if not supabase_service.ensure_session(request.chat_id, user.id, request.query[:60]):
        raise HTTPException(status_code=503, detail="Chat session could not be created")
    response_parts: list[str] = []
    response_sources: list[dict] = []

    async def generate():
        try:
            async for chunk in rag_service.stream_response(request.query, request.document_ids, user.id, request.history):
                data = json.loads(chunk.removeprefix("data: ").strip())
                if "text" in data:
                    response_parts.append(data["text"])
                if "citations" in data:
                    response_sources.extend(data["citations"])
                yield chunk
            supabase_service.save_message(request.user_message_id, request.chat_id, "user", request.query, [], user.id)
            supabase_service.save_message(request.bot_message_id, request.chat_id, "assistant", "".join(response_parts), response_sources, user.id)
        except Exception:
            yield f"data: {json.dumps({'error': 'The assistant is temporarily unavailable.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SessionRequest(BaseModel):
    chat_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=120)


@app.post("/api/v1/sessions")
def save_session(request: SessionRequest, user: CurrentUser = Depends(require_user)):
    if not supabase_service.create_or_update_session(user.id, request.chat_id, request.title, user.id):
        raise HTTPException(status_code=503, detail="Session could not be saved")
    return {"status": "ok", "chat_id": request.chat_id}


@app.get("/api/v1/sessions")
def list_sessions(user: CurrentUser = Depends(require_user)):
    return supabase_service.list_sessions(user.id, user.id)


@app.delete("/api/v1/sessions/{chat_id}")
def delete_session(chat_id: str, user: CurrentUser = Depends(require_user)):
    if not supabase_service.delete_session(chat_id, user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.APP_ENV != "production")
