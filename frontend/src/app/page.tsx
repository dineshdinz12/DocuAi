"use client";

import { useState, useEffect } from "react";
import { DocumentUpload } from "@/components/DocumentUpload";
import { ChatInterface, Message } from "@/components/ChatInterface";
import { AuthModal } from "@/components/AuthModal";
import { FileText, Trash2, ExternalLink, LogOut, LogIn, Plus, MessageSquare, Pencil, Check, Menu, X } from "lucide-react";
import { getSessionId, getCurrentUser, setAuthenticatedUser, clearUserSession, UserProfile } from "@/utils/session";
import { supabase } from "@/lib/supabase";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export interface Document {
  id: string;
  name: string;
  key: string;
  size: number;
  uploaded_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  // Chat Sessions & History Management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("current");
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  useEffect(() => {
    setUser(getCurrentUser());

    // Only listen for Supabase magic link callbacks (not Google — that's handled by Firebase)
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        // Only handle Supabase-specific events (magic link OTP)
        // Google auth is done via Firebase and sets localStorage directly
        if (session?.user && event === "SIGNED_IN" && !getCurrentUser()) {
          const u = session.user;
          const profile: UserProfile = {
            id: u.id,
            name: u.user_metadata?.full_name || u.user_metadata?.name || u.email?.split("@")[0] || "User",
            target: u.email || undefined,
            auth_type: "email",
            avatar_url: u.user_metadata?.avatar_url || undefined,
          };
          setAuthenticatedUser(profile);
          setUser(profile);
        }
      }
    );
    return () => subscription.unsubscribe();
  }, []);

  const fetchDocuments = async (retries = 3) => {
    const currentUser = getCurrentUser();
    try {
      const headers: Record<string, string> = {
        "x-session-id": getSessionId()
      };
      if (currentUser?.id) headers["x-user-id"] = currentUser.id;

      const res = await fetch(`${API_BASE_URL}/api/v1/documents`, { headers });
      if (!res.ok) {
        if (retries > 0) {
          setTimeout(() => fetchDocuments(retries - 1), 1000);
          return;
        }
        console.warn(`[DocuAI] Document list fetch returned status ${res.status}`);
        return;
      }
      const data = await res.json();
      if (Array.isArray(data)) {
        setDocuments(data);
      }
    } catch (e) {
      if (retries > 0) {
        setTimeout(() => fetchDocuments(retries - 1), 1000);
        return;
      }
      console.error("Failed to fetch documents", e);
    }
  };

  const fetchSessions = async () => {
    const currentUser = getCurrentUser();
    try {
      const headers: Record<string, string> = {
        "x-session-id": getSessionId()
      };
      if (currentUser?.id) headers["x-user-id"] = currentUser.id;

      const res = await fetch(`${API_BASE_URL}/api/v1/sessions`, { headers });
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        // Transform Supabase shape to ChatSession shape
        const transformed: ChatSession[] = data.map((s: any) => ({
          id: s.id,
          title: s.title || "Untitled Chat",
          messages: (s.chat_messages || []).map((m: any) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
            sources: m.sources || [],
          })),
          createdAt: s.created_at,
        }));
        setSessions(transformed);
        return;
      }
    } catch (e) {
      console.warn("[Sessions] Supabase fetch failed, falling back to localStorage", e);
    }
    // Fallback: localStorage
    const sessionKey = getSessionId();
    const savedSessions = localStorage.getItem(`docuai_chat_sessions_${sessionKey}`);
    if (savedSessions) {
      try { setSessions(JSON.parse(savedSessions)); } catch { setSessions([]); }
    }
  };

  useEffect(() => {
    fetchDocuments();
    fetchSessions();

    const sessionKey = getSessionId();
    // Load current draft from localStorage
    const currentDraft = localStorage.getItem(`docuai_current_chat_${sessionKey}`);
    if (currentDraft) {
      try { setCurrentMessages(JSON.parse(currentDraft)); } catch { setCurrentMessages([]); }
    } else {
      setCurrentMessages([]);
    }
    setActiveSessionId("current");
  }, [user]);

  // Sync currentMessages back to the sessions list or current draft
  useEffect(() => {
    if (isStreaming) return; // Bypass state/localStorage updates during active streaming to prevent render depth issues
    
    const sessionKey = getSessionId();
    if (activeSessionId === "current") {
      if (currentMessages.length > 0) {
        localStorage.setItem(`docuai_current_chat_${sessionKey}`, JSON.stringify(currentMessages));
      } else {
        localStorage.removeItem(`docuai_current_chat_${sessionKey}`);
      }
    } else if (activeSessionId) {
      setSessions(prev => {
        const updated = prev.map(s => {
          if (s.id === activeSessionId) {
            return { ...s, messages: currentMessages };
          }
          return s;
        });
        localStorage.setItem(`docuai_chat_sessions_${sessionKey}`, JSON.stringify(updated));
        return updated;
      });
    }
  }, [currentMessages, activeSessionId, isStreaming]);

  // Save current conversation to history when starting a New Chat
  const handleNewChat = async () => {
    if (currentMessages.length > 0) {
      const firstUserMsg = currentMessages.find(m => m.role === "user");
      const title = firstUserMsg ? firstUserMsg.content.slice(0, 40) + "..." : "Previous Chat";
      const newChatId = Date.now().toString();

      const newSession: ChatSession = {
        id: newChatId,
        title,
        messages: currentMessages,
        createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setSessions(prev => {
        const updated = [newSession, ...prev];
        localStorage.setItem(`docuai_chat_sessions_${getSessionId()}`, JSON.stringify(updated));
        return updated;
      });

      // Persist session to Supabase
      const currentUser = getCurrentUser();
      try {
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
          "x-session-id": getSessionId(),
        };
        if (currentUser?.id) headers["x-user-id"] = currentUser.id;

        await fetch(`${API_BASE_URL}/api/v1/sessions`, {
          method: "POST",
          headers,
          body: JSON.stringify({ chat_id: newChatId, title }),
        });
      } catch (e) {
        console.warn("[Sessions] Failed to persist session to Supabase", e);
      }
    }

    setCurrentMessages([]);
    localStorage.removeItem(`docuai_current_chat_${getSessionId()}`);
    setActiveSessionId("current");
  };

  // Switch to a past chat session from history
  const handleSelectSession = (session: ChatSession) => {
    setActiveSessionId(session.id);
    setCurrentMessages(session.messages);
  };

  // Delete a past chat session
  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSessions(prev => {
      const updated = prev.filter(s => s.id !== sessionId);
      localStorage.setItem(`docuai_chat_sessions_${getSessionId()}`, JSON.stringify(updated));
      return updated;
    });
    // If the deleted session was active, reset to current
    if (activeSessionId === sessionId) {
      setActiveSessionId("current");
      setCurrentMessages([]);
    }
    // Delete from Supabase too
    try {
      await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}`, {
        method: "DELETE",
        headers: { "x-session-id": getSessionId() },
      });
    } catch (e) {
      console.warn("[Sessions] Supabase delete failed silently", e);
    }
  };

  // Rename a chat session inline
  const startRename = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingSessionId(session.id);
    setRenameValue(session.title);
  };

  const commitRename = async (sessionId: string) => {
    const newTitle = renameValue.trim();
    if (!newTitle) { setRenamingSessionId(null); return; }

    setSessions(prev => {
      const updated = prev.map(s => s.id === sessionId ? { ...s, title: newTitle } : s);
      localStorage.setItem(`docuai_chat_sessions_${getSessionId()}`, JSON.stringify(updated));
      return updated;
    });
    setRenamingSessionId(null);

    // Persist to Supabase
    try {
      await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-session-id": getSessionId() },
        body: JSON.stringify({ chat_id: sessionId, title: newTitle }),
      });
    } catch (e) {
      console.warn("[Sessions] Rename sync to Supabase failed silently", e);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents/${id}`, {
        method: "DELETE",
        headers: { "x-session-id": getSessionId() }
      });
      if (res.ok) {
        setDocuments(docs => docs.filter(d => d.id !== id));
        setSelectedDocIds(ids => ids.filter(selectedId => selectedId !== id));
      }
    } catch (err) {
      console.error("Failed to delete", err);
    }
  };

  const handleView = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    // Use the /view endpoint which serves PDF with Content-Disposition: inline
    // so the browser opens it instead of downloading
    const sessionId = getSessionId();
    const viewUrl = `${API_BASE_URL}/api/v1/documents/${id}/view?session_id=${encodeURIComponent(sessionId)}`;
    window.open(viewUrl, "_blank", "noopener,noreferrer");
  };

  const toggleSelection = (id: string) => {
    setSelectedDocIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const renderSidebarContent = (isMobile: boolean = false) => (
    <>
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-200/50 space-y-4 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <img
              src="/logo-transparent.png"
              alt="DocuAI Logo"
              className="h-10 w-auto object-contain flex-shrink-0"
            />
            <div className="min-w-0">
              <span className="text-[11px] font-semibold text-slate-500 tracking-tight block truncate">
                Knowledge Instantly Retrieved
              </span>
            </div>
          </div>
          {isMobile && (
            <button
              onClick={() => setIsMobileSidebarOpen(false)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors cursor-pointer"
              title="Close menu"
            >
              <X className="w-5 h-5" strokeWidth={1.5} />
            </button>
          )}
        </div>

        {/* New Chat Button */}
        <button
          onClick={() => {
            handleNewChat();
            if (isMobile) setIsMobileSidebarOpen(false);
          }}
          className="w-full flex items-center justify-between py-2.5 px-3 bg-white hover:bg-slate-50 border border-slate-200/80 rounded-xl text-xs font-semibold text-slate-700 transition-all shadow-xs group active:scale-[0.98] cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <Plus className="w-3.5 h-3.5 text-slate-500 group-hover:text-slate-900" strokeWidth={2} />
            <span>New Chat</span>
          </span>
          <span className="text-[10px] text-slate-400 font-mono">⌘N</span>
        </button>
      </div>

      {/* Sidebar Scrollable Content */}
      <div className="flex-1 overflow-y-auto no-scrollbar p-4 space-y-6">

        {/* Recent Chats */}
        {sessions.length > 0 && (
          <div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-1 block mb-2">
              Recent Chats
            </span>
            <div className="space-y-1">
              {sessions.map(session => (
                <div
                  key={session.id}
                  onClick={() => {
                    if (renamingSessionId !== session.id) {
                      handleSelectSession(session);
                      if (isMobile) setIsMobileSidebarOpen(false);
                    }
                  }}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold cursor-pointer transition-all duration-150 group ${activeSessionId === session.id
                    ? "bg-slate-900 text-white shadow-sm"
                    : "text-slate-600 hover:bg-slate-200/40 hover:text-slate-900"
                    }`}
                >
                  <MessageSquare className={`w-3.5 h-3.5 flex-shrink-0 ${activeSessionId === session.id ? "text-slate-300" : "text-slate-400"}`} strokeWidth={1.5} />

                  {/* Title or Inline Rename Input */}
                  {renamingSessionId === session.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onBlur={() => commitRename(session.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename(session.id);
                        if (e.key === "Escape") setRenamingSessionId(null);
                      }}
                      className="flex-1 min-w-0 bg-transparent border-b border-slate-400 outline-none text-xs font-semibold text-white placeholder:text-slate-400 py-0"
                      style={{ color: activeSessionId === session.id ? "white" : "#1e293b" }}
                    />
                  ) : (
                    <span className="truncate flex-1">{session.title}</span>
                  )}

                  {/* Action buttons — appear on hover */}
                  <span className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0">
                    <button
                      onClick={(e) => startRename(session, e)}
                      className={`p-0.5 rounded ${
                        activeSessionId === session.id
                          ? "hover:bg-slate-700 text-slate-400 hover:text-slate-200"
                          : "hover:bg-slate-200 text-slate-400 hover:text-slate-700"
                      }`}
                      title="Rename chat"
                    >
                      <Pencil className="w-3 h-3" strokeWidth={1.5} />
                    </button>
                    <button
                      onClick={(e) => handleDeleteSession(session.id, e)}
                      className={`p-0.5 rounded ${
                        activeSessionId === session.id
                          ? "hover:bg-slate-700 text-slate-400 hover:text-slate-200"
                          : "hover:bg-rose-100 text-slate-400 hover:text-rose-600"
                      }`}
                      title="Delete chat"
                    >
                      <Trash2 className="w-3 h-3" strokeWidth={1.5} />
                    </button>
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Documents Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between px-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Documents</span>
            <span className="text-[10px] font-bold text-slate-600 bg-slate-200/60 px-1.5 py-0.5 rounded-md">
              {documents.length}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 leading-relaxed px-1">
            Select one or multiple documents to query. Unselected files will be ignored.
          </p>

          {/* Clean Upload PDF Trigger */}
          <div>
            <DocumentUpload onUploadSuccess={fetchDocuments} variant="button" />
          </div>

          <div className="space-y-1">
            {documents.length === 0 ? (
              <div className="p-4 text-center border border-dashed border-slate-200/80 rounded-xl bg-white/50">
                <FileText className="w-4 h-4 text-slate-300 mx-auto mb-1.5" strokeWidth={1.5} />
                <p className="text-xs text-slate-500 font-semibold">No PDFs uploaded</p>
                <p className="text-[10px] text-slate-400 mt-0.5">Drag a PDF above to upload</p>
              </div>
            ) : (
              documents.map(doc => (
                <div
                  key={doc.id}
                  onClick={() => toggleSelection(doc.id)}
                  className={`flex items-center justify-between px-3 py-2 rounded-xl cursor-pointer text-xs font-semibold transition-all duration-150 group ${selectedDocIds.includes(doc.id)
                    ? "bg-slate-900 text-white shadow-sm"
                    : "text-slate-600 hover:bg-slate-200/40 hover:text-slate-900"
                    }`}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className={`w-3.5 h-3.5 flex-shrink-0 ${selectedDocIds.includes(doc.id) ? "text-slate-300" : "text-slate-400"}`} strokeWidth={1.5} />
                    <span className="truncate">{doc.name}</span>
                  </div>

                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => handleView(doc.id, e)}
                      className={`p-1 rounded ${selectedDocIds.includes(doc.id) ? "hover:bg-slate-800 text-slate-300" : "hover:bg-slate-200 text-slate-600"}`}
                      title="View PDF"
                    >
                      <ExternalLink className="w-3 h-3" strokeWidth={1.5} />
                    </button>
                    <button
                      onClick={(e) => handleDelete(doc.id, e)}
                      className={`p-1 rounded ${selectedDocIds.includes(doc.id) ? "hover:bg-slate-800 text-slate-300" : "hover:bg-rose-100 text-rose-600"}`}
                      title="Delete"
                    >
                      <Trash2 className="w-3 h-3" strokeWidth={1.5} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* User Account Footer */}
      <div className="p-4 border-t border-slate-200/50 bg-slate-50 flex-shrink-0">
        {user ? (
          <div className="flex items-center justify-between p-2 bg-white rounded-xl border border-slate-200/70 shadow-xs">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-7 h-7 rounded-lg bg-slate-900 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 overflow-hidden">
                {user.avatar_url ? (
                  <img src={user.avatar_url} alt={user.name} className="w-full h-full object-cover" />
                ) : (
                  user.name.charAt(0).toUpperCase()
                )}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-bold text-slate-900 truncate">{user.name}</p>
                <p className="text-[10px] text-slate-400 truncate">Google User</p>
              </div>
            </div>

            <button
              onClick={() => {
                clearUserSession();
                setUser(null);
                fetchDocuments();
                if (isMobile) setIsMobileSidebarOpen(false);
              }}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
              title="Sign Out"
            >
              <LogOut className="w-3.5 h-3.5" strokeWidth={1.5} />
            </button>
          </div>
        ) : (
          <button
            onClick={() => {
              setIsAuthModalOpen(true);
              if (isMobile) setIsMobileSidebarOpen(false);
            }}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-3 bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs rounded-xl transition-all shadow-sm active:scale-[0.98] cursor-pointer"
          >
            <LogIn className="w-3.5 h-3.5" strokeWidth={1.5} />
            <span>Sign In with Google</span>
          </button>
        )}
      </div>
    </>
  );

  return (
    <div className="flex h-screen bg-[#FFFFFF] text-slate-900 font-sans antialiased selection:bg-slate-900 selection:text-white">

      {/* Premium wider Desktop Sidebar */}
      <aside className="w-[280px] bg-slate-50 border-r border-slate-200/60 flex flex-col hidden md:flex h-full select-none">
        {renderSidebarContent(false)}
      </aside>

      {/* Mobile Sidebar Overlay & Drawer */}
      {isMobileSidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
            onClick={() => setIsMobileSidebarOpen(false)}
          />
          {/* Slide-out Drawer */}
          <aside className="relative w-[280px] sm:w-[320px] max-w-[85vw] bg-slate-50 h-full flex flex-col z-50 shadow-2xl animate-in slide-in-from-left duration-200 select-none">
            {renderSidebarContent(true)}
          </aside>
        </div>
      )}

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col h-full bg-white relative min-w-0">
        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between p-3.5 border-b border-slate-200 bg-white sticky top-0 z-10">
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsMobileSidebarOpen(true)}
              className="p-1.5 border border-slate-200 hover:bg-slate-100 rounded-xl text-slate-700 transition-colors flex items-center justify-center cursor-pointer"
              title="Open menu & history"
              aria-label="Open menu"
            >
              <Menu className="w-5 h-5" strokeWidth={1.75} />
            </button>
            <img
              src="/logo-transparent.png"
              alt="DocuAI Logo"
              className="h-8 w-auto object-contain"
            />
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleNewChat}
              className="p-1.5 border border-slate-200 rounded-lg text-slate-700 font-medium text-xs flex items-center gap-1 cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New</span>
            </button>
            {!user && (
              <button
                onClick={() => setIsAuthModalOpen(true)}
                className="py-1 px-3 bg-slate-900 text-white font-semibold text-xs rounded-lg shadow-sm cursor-pointer"
              >
                Sign In
              </button>
            )}
          </div>
        </header>

        {/* Claude / ChatGPT Style Streaming Chat Interface */}
        <div className="flex-1 overflow-hidden">
          <ChatInterface
            selectedDocuments={documents.filter(d => selectedDocIds.includes(d.id))}
            messages={currentMessages}
            setMessages={setCurrentMessages}
            onUploadSuccess={fetchDocuments}
            isStreaming={isStreaming}
            setIsStreaming={setIsStreaming}
          />
        </div>
      </main>

      {/* Google Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onSuccess={(authenticatedUser) => {
          setUser(authenticatedUser);
          fetchDocuments();
        }}
      />
    </div>
  );
}
