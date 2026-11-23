"use client";

import { Dispatch, FormEvent, SetStateAction, useEffect, useRef, useState } from "react";
import { ArrowUp, Check, Copy, FileText, Loader2, Sparkles } from "lucide-react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiFetch } from "@/lib/api";
import { DocumentUpload } from "./DocumentUpload";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ name: string; page?: number | null; document_id?: string }>;
}

interface SelectedDocument { id: string; name: string }
interface Props {
  chatId: string;
  selectedDocuments: SelectedDocument[];
  messages: Message[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
  onUploadSuccess: () => void;
  onChatSaved: () => void;
  onOpenDocument: (documentId: string) => void;
}

const starters = [
  "Summarize the key decisions",
  "Find risks and unresolved questions",
  "Compare the selected documents",
  "Create an executive brief",
];

export function ChatInterface({ chatId, selectedDocuments, messages, setMessages, onUploadSuccess, onChatSaved, onOpenDocument }: Props) {
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  const send = async (value = input) => {
    const query = value.trim();
    if (!query || streaming) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: query };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [...current, userMessage, { id: assistantId, role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);
    try {
      const response = await apiFetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          document_ids: selectedDocuments.map((document) => document.id),
          history: messages.map(({ role, content }) => ({ role, content })),
          chat_id: chatId,
          user_message_id: userMessage.id,
          bot_message_id: assistantId,
        }),
      });
      if (!response.body) throw new Error("Response stream is unavailable");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value: bytes } = await reader.read();
        if (done) break;
        buffer += decoder.decode(bytes, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";
        for (const event of events) {
          if (!event.startsWith("data: ")) continue;
          const data = JSON.parse(event.slice(6));
          if (data.error) throw new Error(data.error);
          setMessages((current) => current.map((message) => message.id === assistantId ? {
            ...message,
            content: data.text ? message.content + data.text : message.content,
            sources: data.citations || message.sources,
          } : message));
        }
      }
      onChatSaved();
    } catch (error) {
      const message = error instanceof Error ? error.message : "The assistant is unavailable.";
      setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: `I couldn't complete that request. ${message}` } : item));
    } finally {
      setStreaming(false);
    }
  };

  const submit = (event: FormEvent) => { event.preventDefault(); void send(); };
  const copy = async (message: Message) => {
    await navigator.clipboard.writeText(message.content);
    setCopied(message.id);
    window.setTimeout(() => setCopied(null), 1500);
  };

  return (
    <section className="flex h-full min-w-0 flex-col bg-white">
      <header className="flex h-16 items-center justify-between border-b border-slate-200/70 bg-white/80 pl-16 pr-5 backdrop-blur-xl md:px-8">
        <div>
          <h1 className="text-sm font-semibold text-slate-900">Intelligence workspace</h1>
          <p className="mt-0.5 text-xs text-slate-500">{selectedDocuments.length ? `${selectedDocuments.length} source${selectedDocuments.length > 1 ? "s" : ""} active` : "General conversation"}</p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col px-4 py-8 md:px-8">
          {messages.length === 0 ? (
            <div className="mx-auto my-auto w-full max-w-2xl py-12 text-center">
              <Image src="/logo-transparent.png" alt="DocuAI" width={140} height={62} className="mx-auto h-14 w-auto object-contain" />
              <h2 className="mt-5 text-3xl font-semibold tracking-tight text-slate-950">How can I help you today?</h2>
              <p className="mt-3 text-sm text-slate-600">Ask a question or select PDFs to get answers with sources.</p>
              <div className="mt-8 grid gap-3 sm:grid-cols-2">
                {starters.map((starter) => <button key={starter} onClick={() => void send(starter)} className="rounded-2xl border border-slate-200 bg-white p-4 text-left text-sm font-medium text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:border-violet-200 hover:shadow-md">{starter}</button>)}
              </div>
            </div>
          ) : (
            <div className="space-y-8 pb-6">
              {messages.map((message) => (
                <article key={message.id} className={message.role === "user" ? "ml-auto max-w-2xl" : "max-w-3xl"}>
                  {message.role === "user" ? (
                    <div className="rounded-[22px] rounded-br-md bg-slate-900 px-5 py-3.5 text-[15px] leading-6 text-white shadow-sm">{message.content}</div>
                  ) : (
                    <div className="group flex gap-4">
                      <div className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-violet-100 text-violet-600"><Sparkles className="h-4 w-4" /></div>
                      <div className="min-w-0 flex-1">
                        {!message.content && streaming ? <div className="flex items-center gap-2 py-2 text-sm text-slate-500"><Loader2 className="h-4 w-4 animate-spin text-violet-500" />Reasoning across your sources…</div> : <div className="prose prose-slate max-w-none text-[15px] leading-7"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>}
                        {!!message.sources?.length && <div className="mt-4 flex flex-wrap gap-2">{message.sources.map((source, index) => <button type="button" disabled={!source.document_id} onClick={() => source.document_id && onOpenDocument(source.document_id)} key={`${source.name}-${source.page}-${index}`} className="inline-flex items-center gap-1.5 rounded-lg border border-violet-100 bg-violet-50 px-2.5 py-1.5 text-xs font-medium text-violet-700 transition enabled:hover:border-violet-300 enabled:hover:bg-violet-100 disabled:cursor-default"><FileText className="h-3.5 w-3.5" />{source.name}{source.page != null ? ` · p.${source.page + 1}` : ""}</button>)}</div>}
                        {message.content && <button onClick={() => void copy(message)} className="mt-3 flex items-center gap-1.5 text-xs text-slate-600 transition hover:text-slate-900">{copied === message.id ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}{copied === message.id ? "Copied" : "Copy"}</button>}
                      </div>
                    </div>
                  )}
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-slate-200/60 bg-white/85 p-4 backdrop-blur-xl">
        <form onSubmit={submit} className="mx-auto max-w-4xl rounded-[22px] border border-slate-200 bg-white p-2 shadow-[0_12px_40px_rgba(15,23,42,0.08)] transition focus-within:border-violet-300 focus-within:ring-4 focus-within:ring-violet-100/60">
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }} rows={2} placeholder={selectedDocuments.length ? "Ask a question across the selected sources…" : "Ask anything, or select documents for grounded answers…"} className="block max-h-40 w-full resize-none bg-transparent px-3 py-2 text-[15px] leading-6 text-slate-900 outline-none placeholder:text-slate-400" />
          <div className="flex items-center justify-between px-1 pb-1">
            <div className="flex items-center gap-2"><DocumentUpload onUploadSuccess={onUploadSuccess} variant="icon" /><span className="hidden text-xs text-slate-400 sm:inline">PDF up to 25 MB</span></div>
            <button disabled={!input.trim() || streaming} className="grid h-9 w-9 place-items-center rounded-xl bg-slate-900 text-white transition hover:bg-violet-600 disabled:bg-slate-200 disabled:text-slate-400" aria-label="Send message"><ArrowUp className="h-4 w-4" /></button>
          </div>
        </form>
        <p className="mx-auto mt-2 max-w-4xl text-center text-[11px] text-slate-400">Verify important information against the cited source.</p>
      </div>
    </section>
  );
}
