"use client";

import { useCallback, useEffect, useState } from "react";
import Image from "next/image";
import { FileText, LogOut, Menu, MessageSquare, PanelLeftClose, Pencil, Plus, Search, Trash2, UserRound, X } from "lucide-react";
import type { User } from "@supabase/supabase-js";
import { AuthModal } from "@/components/AuthModal";
import { ChatInterface, Message } from "@/components/ChatInterface";
import { DocumentUpload } from "@/components/DocumentUpload";
import { apiFetch } from "@/lib/api";
import { supabase } from "@/lib/supabase";

interface DocumentRecord { id: string; name: string; size: number; status?: "processing" | "ready" | "failed"; uploaded_at: string }
interface SessionRecord { id: string; title: string; messages: Message[]; createdAt: string }
interface ApiMessage { id: string; role: "user" | "assistant"; content: string; sources?: Message["sources"]; created_at: string }
interface ApiSession { id: string; title?: string; created_at: string; chat_messages?: ApiMessage[] }

const newId = () => crypto.randomUUID();

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [activeId, setActiveId] = useState(newId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [search, setSearch] = useState("");

  const loadDocuments = useCallback(async () => {
    try {
      const response = await apiFetch("/api/v1/documents");
      setDocuments(await response.json());
    } catch { setDocuments([]); }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      const response = await apiFetch("/api/v1/sessions");
      const data: ApiSession[] = await response.json();
      setSessions(data.map((session) => ({
        id: session.id,
        title: session.title || "New conversation",
        createdAt: session.created_at,
        messages: (session.chat_messages || []).sort((a, b) => a.created_at.localeCompare(b.created_at)).map(({ id, role, content, sources }) => ({ id, role, content, sources: sources || [] })),
      })));
    } catch { setSessions([]); }
  }, []);

  useEffect(() => {
    void supabase.auth.getUser().then(({ data }) => {
      setUser(data.user);
      setAuthReady(true);
      setAuthOpen(!data.user);
      if (data.user) { void loadDocuments(); void loadSessions(); }
    });
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
      setAuthReady(true);
      setAuthOpen(!session?.user);
      if (session?.user) { void loadDocuments(); void loadSessions(); }
    });
    return () => data.subscription.unsubscribe();
  }, [loadDocuments, loadSessions]);

  useEffect(() => {
    if (!documents.some((document) => document.status === "processing")) return;
    const timer = window.setInterval(() => void loadDocuments(), 3000);
    return () => window.clearInterval(timer);
  }, [documents, loadDocuments]);

  const newChat = () => { setActiveId(newId()); setMessages([]); setSidebarOpen(false); };
  const selectSession = (session: SessionRecord) => { setActiveId(session.id); setMessages(session.messages); setSidebarOpen(false); };

  const deleteSession = async (id: string) => {
    await apiFetch(`/api/v1/sessions/${id}`, { method: "DELETE" });
    setSessions((current) => current.filter((session) => session.id !== id));
    if (activeId === id) newChat();
  };

  const renameSession = async (session: SessionRecord) => {
    const title = window.prompt("Rename conversation", session.title)?.trim();
    if (!title || title === session.title) return;
    await apiFetch("/api/v1/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ chat_id: session.id, title }) });
    setSessions((current) => current.map((item) => item.id === session.id ? { ...item, title } : item));
  };

  const deleteDocument = async (id: string) => {
    await apiFetch(`/api/v1/documents/${id}`, { method: "DELETE" });
    setDocuments((current) => current.filter((document) => document.id !== id));
    setSelectedIds((current) => current.filter((selected) => selected !== id));
  };

  const viewDocument = async (id: string) => {
    const response = await apiFetch(`/api/v1/documents/${id}/content`);
    const url = URL.createObjectURL(await response.blob());
    window.open(url, "_blank", "noopener,noreferrer");
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };

  const signOut = async () => { await supabase.auth.signOut(); setDocuments([]); setSessions([]); newChat(); };
  const selectedDocuments = documents.filter((document) => selectedIds.includes(document.id));
  const filteredSessions = sessions.filter((session) => session.title.toLowerCase().includes(search.toLowerCase()));

  const sidebar = (
    <aside className="flex h-full w-[300px] flex-col border-r border-white/10 bg-[#0c1220] text-slate-200">
      <div className="flex h-16 items-center justify-between border-b border-white/10 px-5">
        <div className="flex items-center gap-3"><div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-violet-500 to-indigo-500 font-bold text-white">D</div><div><p className="font-semibold text-white">DocuAI</p><p className="text-[11px] text-slate-500">Document intelligence</p></div></div>
        <button onClick={() => setSidebarOpen(false)} className="rounded-lg p-2 text-slate-500 hover:bg-white/10 hover:text-white md:hidden" aria-label="Close sidebar"><X className="h-4 w-4" /></button>
      </div>
      <div className="p-4"><button onClick={newChat} className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-white font-medium text-slate-950 transition hover:bg-violet-50"><Plus className="h-4 w-4" />New conversation</button></div>
      <div className="flex-1 overflow-y-auto px-3 pb-4">
        <div className="relative mb-3"><Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-600" /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search conversations" className="h-9 w-full rounded-lg border border-white/10 bg-white/5 pl-9 pr-3 text-xs outline-none placeholder:text-slate-600 focus:border-violet-500/50" /></div>
        <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-600">Recent</p>
        <div className="space-y-1">{filteredSessions.map((session) => <button key={session.id} onClick={() => selectSession(session)} className={`group flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm transition ${activeId === session.id ? "bg-violet-500/15 text-violet-200" : "text-slate-400 hover:bg-white/5 hover:text-white"}`}><MessageSquare className="h-4 w-4 shrink-0" /><span className="min-w-0 flex-1 truncate">{session.title}</span><span onClick={(event) => { event.stopPropagation(); void renameSession(session); }} className="rounded p-1 opacity-0 hover:bg-white/10 hover:text-white group-hover:opacity-100"><Pencil className="h-3.5 w-3.5" /></span><span onClick={(event) => { event.stopPropagation(); void deleteSession(session.id); }} className="rounded p-1 opacity-0 hover:bg-rose-500/10 hover:text-rose-300 group-hover:opacity-100"><Trash2 className="h-3.5 w-3.5" /></span></button>)}</div>
        <div className="my-5 h-px bg-white/10" />
        <div className="mb-3 flex items-center justify-between px-2"><p className="text-[11px] font-semibold uppercase tracking-wider text-slate-600">Sources</p><span className="text-xs text-slate-600">{documents.length}</span></div>
        <DocumentUpload onUploadSuccess={loadDocuments} />
        <div className="mt-3 space-y-1">{documents.map((document) => <div key={document.id} className={`group flex items-center gap-2 rounded-xl border px-3 py-2.5 transition ${document.status === "processing" ? "cursor-wait opacity-65" : "cursor-pointer"} ${selectedIds.includes(document.id) ? "border-violet-500/30 bg-violet-500/15 text-violet-100" : "border-transparent text-slate-400 hover:bg-white/5 hover:text-white"}`} onClick={() => document.status !== "processing" && document.status !== "failed" && setSelectedIds((current) => current.includes(document.id) ? current.filter((id) => id !== document.id) : [...current, document.id])}><FileText className="h-4 w-4 shrink-0" /><span className="min-w-0 flex-1 truncate text-xs font-medium">{document.name}</span>{document.status && document.status !== "ready" && <span className={`text-[10px] ${document.status === "failed" ? "text-rose-300" : "text-amber-300"}`}>{document.status}</span>}<button onClick={(event) => { event.stopPropagation(); void viewDocument(document.id); }} className="opacity-0 group-hover:opacity-100" title="Open PDF"><PanelLeftClose className="h-3.5 w-3.5" /></button><button onClick={(event) => { event.stopPropagation(); void deleteDocument(document.id); }} className="opacity-0 hover:text-rose-300 group-hover:opacity-100" title="Delete PDF"><Trash2 className="h-3.5 w-3.5" /></button></div>)}</div>
      </div>
      <div className="border-t border-white/10 p-4"><div className="flex items-center gap-3"><div className="grid h-9 w-9 place-items-center overflow-hidden rounded-full bg-white/10">{user?.user_metadata?.avatar_url ? <Image src={user.user_metadata.avatar_url} alt="Account avatar" width={36} height={36} unoptimized className="h-full w-full object-cover" /> : <UserRound className="h-4 w-4" />}</div><div className="min-w-0 flex-1"><p className="truncate text-sm font-medium text-white">{user?.user_metadata?.full_name || user?.email?.split("@")[0]}</p><p className="truncate text-[11px] text-slate-500">{user?.email}</p></div><button onClick={() => void signOut()} className="rounded-lg p-2 text-slate-500 hover:bg-white/10 hover:text-white" aria-label="Sign out"><LogOut className="h-4 w-4" /></button></div></div>
    </aside>
  );

  if (!authReady) return <div className="grid h-screen place-items-center bg-[#0c1220] text-sm text-slate-400">Preparing your workspace…</div>;

  return (
    <div className="flex h-dvh overflow-hidden bg-[#f7f8fc]">
      <div className="hidden md:block">{sidebar}</div>
      {sidebarOpen && <div className="fixed inset-0 z-40 flex md:hidden"><button className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar overlay" /> <div className="relative z-10">{sidebar}</div></div>}
      <main className="flex min-w-0 flex-1 flex-col"><button onClick={() => setSidebarOpen(true)} className="absolute left-4 top-4 z-20 rounded-xl border border-slate-200 bg-white p-2 text-slate-600 shadow-sm md:hidden" aria-label="Open sidebar"><Menu className="h-4 w-4" /></button><ChatInterface chatId={activeId} selectedDocuments={selectedDocuments} messages={messages} setMessages={setMessages} onUploadSuccess={loadDocuments} onChatSaved={loadSessions} onOpenDocument={(id) => void viewDocument(id)} /></main>
      <AuthModal isOpen={authOpen} onClose={() => user && setAuthOpen(false)} />
    </div>
  );
}
