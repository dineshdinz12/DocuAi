import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import Client, create_client

from core.config import settings

logger = logging.getLogger(__name__)
_client: Optional[Client] = None


def get_supabase() -> Optional[Client]:
    global _client
    if _client is None and settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        try:
            _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        except Exception:
            logger.exception("Supabase client initialization failed")
    return _client


class SupabaseService:
    def is_configured(self) -> bool:
        return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY)

    def save_document(self, doc_id: str, name: str, storage_key: str, size: int, session_id: str, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            client.table("documents").upsert({"id": doc_id, "user_id": user_id, "session_id": session_id, "name": name, "storage_key": storage_key, "size": size, "status": "processing"}).execute()
            return True
        except Exception:
            logger.exception("Unable to save document metadata")
            return False

    def list_documents(self, session_id: str, user_id: str):
        client = get_supabase()
        if not client:
            return None
        try:
            return client.table("documents").select("id,name,size,status,uploaded_at,storage_key").eq("user_id", user_id).order("uploaded_at", desc=True).execute().data or []
        except Exception:
            logger.exception("Unable to list documents")
            return None

    def get_document(self, doc_id: str, user_id: str):
        client = get_supabase()
        if not client:
            return None
        try:
            rows = client.table("documents").select("*").eq("id", doc_id).eq("user_id", user_id).limit(1).execute().data
            return rows[0] if rows else None
        except Exception:
            logger.exception("Unable to read document")
            return None

    def user_owns_documents(self, doc_ids: list[str], user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            rows = client.table("documents").select("id").eq("user_id", user_id).in_("id", doc_ids).execute().data or []
            return {row["id"] for row in rows} == set(doc_ids)
        except Exception:
            logger.exception("Unable to validate document ownership")
            return False

    def delete_document(self, doc_id: str, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            client.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
            return True
        except Exception:
            logger.exception("Unable to delete document metadata")
            return False

    def set_document_status(self, doc_id: str, user_id: str, status: str) -> bool:
        client = get_supabase()
        if not client or status not in {"processing", "ready", "failed"}:
            return False
        try:
            client.table("documents").update({"status": status}).eq("id", doc_id).eq("user_id", user_id).execute()
            return True
        except Exception:
            logger.exception("Unable to update document status")
            return False

    def claim_failed_document(self, doc_id: str, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        result = client.table("documents").update({"status": "processing"}).eq(
            "id", doc_id
        ).eq("user_id", user_id).eq("status", "failed").execute()
        return bool(result.data)

    def create_or_update_session(self, session_id: str, chat_id: str, title: str, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            existing = client.table("chat_sessions").select("id,user_id").eq("id", chat_id).limit(1).execute().data or []
            if existing and existing[0]["user_id"] != user_id:
                return False
            client.table("chat_sessions").upsert({"id": chat_id, "user_id": user_id, "session_id": session_id, "title": title, "updated_at": datetime.now(timezone.utc).isoformat()}).execute()
            return True
        except Exception:
            logger.exception("Unable to save chat session")
            return False

    def ensure_session(self, chat_id: str, user_id: str, title: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            existing = client.table("chat_sessions").select("id,user_id").eq("id", chat_id).limit(1).execute().data or []
            if existing:
                return existing[0]["user_id"] == user_id
            return self.create_or_update_session(user_id, chat_id, title or "New chat", user_id)
        except Exception:
            logger.exception("Unable to ensure chat session")
            return False

    def list_sessions(self, session_id: str, user_id: str):
        client = get_supabase()
        if not client:
            return []
        try:
            return client.table("chat_sessions").select("*,chat_messages(*)").eq("user_id", user_id).order("updated_at", desc=True).execute().data or []
        except Exception:
            logger.exception("Unable to list chat sessions")
            return []

    def delete_session(self, chat_id: str, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            rows = client.table("chat_sessions").delete().eq("id", chat_id).eq("user_id", user_id).execute().data
            return bool(rows)
        except Exception:
            logger.exception("Unable to delete chat session")
            return False

    def save_message(self, msg_id: str, chat_id: str, role: str, content: str, sources: list, user_id: str) -> bool:
        client = get_supabase()
        if not client:
            return False
        try:
            owner = client.table("chat_sessions").select("id").eq("id", chat_id).eq("user_id", user_id).limit(1).execute().data
            if not owner:
                return False
            client.table("chat_messages").upsert({"id": msg_id, "session_id": chat_id, "role": role, "content": content, "sources": sources}).execute()
            return True
        except Exception:
            logger.exception("Unable to save chat message")
            return False


supabase_service = SupabaseService()
