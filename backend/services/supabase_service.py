import os
from supabase import create_client, Client
from typing import Optional
from core.config import settings

_client: Optional[Client] = None

def get_supabase() -> Optional[Client]:
    """Return a Supabase client using the service role key (backend only)."""
    global _client
    if _client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            print("[Supabase] Missing SUPABASE_URL or SUPABASE_SERVICE_KEY — persistence disabled.")
            return None
        try:
            _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            print("[Supabase] Connected successfully.")
        except Exception as e:
            print(f"[Supabase] Connection failed: {e}")
            return None
    return _client


class SupabaseService:
    """High-level service for persisting chat sessions, messages and document metadata."""

    # ─── Document Metadata ────────────────────────────────────────────────────

    def save_document(self, doc_id: str, name: str, storage_key: str, size: int,
                      session_id: str, user_id: Optional[str] = None):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("documents").upsert({
                "id": doc_id,
                "user_id": user_id,
                "session_id": session_id,
                "name": name,
                "storage_key": storage_key,
                "size": size,
            }).execute()
        except Exception as e:
            print(f"[Supabase] save_document error: {e}")

    def list_documents(self, session_id: str, user_id: Optional[str] = None):
        sb = get_supabase()
        if not sb:
            return None
        try:
            query = sb.table("documents").select("*")
            if user_id:
                query = query.eq("user_id", user_id)
            else:
                query = query.eq("session_id", session_id)
            res = query.order("uploaded_at", desc=True).execute()
            return res.data if res.data else []
        except Exception as e:
            print(f"[Supabase] list_documents error: {e}")
            return None

    def delete_document(self, doc_id: str, session_id: str):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("documents").delete().eq("id", doc_id).execute()
        except Exception as e:
            print(f"[Supabase] delete_document error: {e}")

    # ─── Chat Sessions ─────────────────────────────────────────────────────────

    def create_or_update_session(self, session_id: str, chat_id: str, title: str,
                                  user_id: Optional[str] = None):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("chat_sessions").upsert({
                "id": chat_id,
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "updated_at": "now()",
            }).execute()
        except Exception as e:
            print(f"[Supabase] create_or_update_session error: {e}")

    def list_sessions(self, session_id: str, user_id: Optional[str] = None):
        sb = get_supabase()
        if not sb:
            return []
        try:
            query = sb.table("chat_sessions").select("*, chat_messages(*)") 
            if user_id:
                query = query.eq("user_id", user_id)
            else:
                query = query.eq("session_id", session_id)
            res = query.order("updated_at", desc=True).execute()
            return res.data if res.data else []
        except Exception as e:
            print(f"[Supabase] list_sessions error: {e}")
            return []

    def delete_session(self, chat_id: str):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("chat_sessions").delete().eq("id", chat_id).execute()
        except Exception as e:
            print(f"[Supabase] delete_session error: {e}")

    # ─── Chat Messages ─────────────────────────────────────────────────────────

    def save_message(self, msg_id: str, chat_id: str, role: str, content: str,
                     sources: list = None):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("chat_messages").upsert({
                "id": msg_id,
                "session_id": chat_id,
                "role": role,
                "content": content,
                "sources": sources or [],
            }).execute()
        except Exception as e:
            print(f"[Supabase] save_message error: {e}")

    def get_messages(self, chat_id: str):
        sb = get_supabase()
        if not sb:
            return []
        try:
            res = sb.table("chat_messages")\
                    .select("*")\
                    .eq("session_id", chat_id)\
                    .order("created_at")\
                    .execute()
            return res.data if res.data else []
        except Exception as e:
            print(f"[Supabase] get_messages error: {e}")
            return []

    # ─── User Profiles ─────────────────────────────────────────────────────────

    def upsert_user_profile(self, user_id: str, email: str, name: str, avatar_url: str = None):
        sb = get_supabase()
        if not sb:
            return
        try:
            sb.table("user_profiles").upsert({
                "id": user_id,
                "email": email,
                "name": name,
                "avatar_url": avatar_url,
            }).execute()
        except Exception as e:
            print(f"[Supabase] upsert_user_profile error: {e}")


supabase_service = SupabaseService()
