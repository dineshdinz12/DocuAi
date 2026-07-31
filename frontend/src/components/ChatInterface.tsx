"use client";

import { useState, useRef, useEffect, Dispatch, SetStateAction } from "react";
import { Send, Copy, Check, ArrowUpRight, FileText, RefreshCw, Mic } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Document } from "@/app/page";
import { getSessionId } from "@/utils/session";
import { DocumentUpload } from "@/components/DocumentUpload";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Array<{ name: string; page?: number | null }>;
}

interface ChatInterfaceProps {
  selectedDocuments: Document[];
  messages: Message[];
  setMessages: Dispatch<SetStateAction<Message[]>>;
  onUploadSuccess?: () => void;
  isStreaming: boolean;
  setIsStreaming: Dispatch<SetStateAction<boolean>>;
}

export function ChatInterface({ 
  selectedDocuments, 
  messages, 
  setMessages, 
  onUploadSuccess,
  isStreaming,
  setIsStreaming
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isListening, setIsListening] = useState(false);
  const [recognition, setRecognition] = useState<any>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recog = new SpeechRecognition();
        recog.continuous = false;
        recog.interimResults = false;
        recog.lang = "en-US";

        recog.onstart = () => {
          setIsListening(true);
        };

        recog.onerror = (event: any) => {
          console.error("Speech recognition error", event.error);
          setIsListening(false);
        };

        recog.onend = () => {
          setIsListening(false);
        };

        recog.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          if (transcript) {
            setInput(prev => (prev ? prev + " " + transcript : transcript));
          }
        };

        setRecognition(recog);
      }
    }
  }, []);

  const toggleListening = () => {
    if (!recognition) {
      alert("Speech recognition is not supported in your browser. Please try Chrome or Safari.");
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  const suggestionCards = selectedDocuments.length > 0 
    ? [
        { title: "Summarize Document", query: `Provide a detailed summary of ${selectedDocuments[0].name}.` },
        { title: "Extract Key Metrics", query: "List all specific metrics, key dates, and statistics in these files." },
        { title: "Main Conclusions", query: "What are the primary conclusions and action items?" },
        { title: "Format as Table", query: "Format the key findings into a clean Markdown table." }
      ]
    : [
        { title: "Document Upload Guide", query: "How do I upload PDFs to start analyzing?" },
        { title: "Platform Overview", query: "Explain how DocuAI handles vector search across documents." },
        { title: "Multi-File Analysis", query: "Can I query multiple documents at the same time?" },
        { title: "Data Security", query: "How is my document data isolated and secured?" }
      ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  const handleSend = async (query: string = input) => {
    if (!query.trim() || isStreaming) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsStreaming(true);

    const botMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, { id: botMessageId, role: "assistant", content: "" }]);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const currentUser = typeof window !== "undefined" 
        ? (() => { try { return JSON.parse(localStorage.getItem("docuai_user") || "null"); } catch { return null; } })()
        : null;
      
      // Generate a stable chat_id for this conversation (reuse or create)
      const chatIdKey = `docuai_active_chat_id_${getSessionId()}`;
      let activeChatId = localStorage.getItem(chatIdKey);
      if (!activeChatId) {
        activeChatId = `chat_${Date.now()}`;
        localStorage.setItem(chatIdKey, activeChatId);
      }

      const headers: Record<string, string> = { 
        "Content-Type": "application/json",
        "x-session-id": getSessionId()
      };
      if (currentUser?.id) headers["x-user-id"] = currentUser.id;

      const res = await fetch(`${baseUrl}/api/v1/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({ 
          query,
          document_ids: selectedDocuments.map(d => d.id),
          history: messages.map(m => ({ role: m.role, content: m.content })),
          chat_id: activeChatId,
          user_message_id: userMessage.id,
          bot_message_id: botMessageId,
        }),
      });
      
       if (!res.body) throw new Error("No response stream");
      
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      
      let textQueue = "";
      let displayedText = "";
      let isStreamFinished = false;

      // Start the typing animation interval loop
      const typingPromise = new Promise<void>((resolve) => {
        const typingInterval = setInterval(() => {
          if (textQueue.length > 0) {
            // Pick a chunk length (dynamic: speed up slightly if queue grows to avoid lag)
            const takeCount = textQueue.length > 60 ? 6 : textQueue.length > 25 ? 3 : 1;
            const chunk = textQueue.slice(0, takeCount);
            textQueue = textQueue.slice(takeCount);
            displayedText += chunk;

            setMessages((prev) => 
              prev.map((msg) => 
                msg.id === botMessageId ? { ...msg, content: displayedText } : msg
              )
            );
          } else if (isStreamFinished) {
            clearInterval(typingInterval);
            resolve();
          }
        }, 30);
      });

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            isStreamFinished = true;
            break;
          }
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              try {
                const jsonStr = trimmed.slice(6);
                const data = JSON.parse(jsonStr);
                if (data.text) {
                  textQueue += data.text;
                } else if (data.citations && Array.isArray(data.citations)) {
                  // Phase A: attach citations to the bot message
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === botMessageId ? { ...msg, sources: data.citations } : msg
                    )
                  );
                }
              } catch (e) {
                console.warn("SSE parse error", e);
              }
            }
          }
        }
        
        // Wait for typing animation to catch up completely
        await typingPromise;
      } catch (error) {
        isStreamFinished = true;
        throw error;
      }
    } catch (error) {
      console.error("Streaming error:", error);
      setMessages((prev) => 
        prev.map((msg) => 
          msg.id === botMessageId 
            ? { ...msg, content: "I encountered an error connecting to the vector streaming engine." } 
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-white relative">
      {/* Sticky Document Context Badge Bar at the top */}
      <div className="w-full border-b border-slate-200/60 bg-slate-50/50 backdrop-blur-xs px-6 py-2.5 flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-2 min-w-0">
          {selectedDocuments.length > 0 ? (
            <>
              <div className="flex items-center gap-1.5 text-slate-700 font-semibold bg-emerald-50 text-emerald-700 px-2.5 py-0.5 rounded-full border border-emerald-200/50 text-[10px] uppercase tracking-wider flex-shrink-0">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                </span>
                <span>Document Mode</span>
              </div>
              <span className="text-slate-300">|</span>
              <span className="font-medium text-slate-600 truncate text-[11px]">
                Querying: <span className="font-semibold text-slate-800">{selectedDocuments.map(d => d.name).join(", ")}</span>
              </span>
            </>
          ) : (
            <div className="flex items-center gap-1.5 text-slate-600 bg-slate-100 px-2.5 py-0.5 rounded-full border border-slate-200/50 font-medium text-[10px] uppercase tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-slate-400" />
              <span>Conversation Mode</span>
            </div>
          )}
        </div>
        
        {selectedDocuments.length > 0 && (
          <span className="text-[10px] text-slate-400 font-medium hidden sm:inline flex-shrink-0">
            Answers are grounded in selected files
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto no-scrollbar px-6 py-8 w-full">
        {messages.length === 0 ? (
          <div className="max-w-3xl mx-auto w-full py-16 space-y-8 animate-in fade-in duration-300">
            <div className="text-center">
              <img 
                src="/logo-transparent.png" 
                alt="DocuAI Logo" 
                className="w-20 h-20 mx-auto mb-4 object-contain" 
              />
              <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
                How can I help you today?
              </h2>
              <p className="mt-2 text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
                Ask complex questions across your uploaded PDF collection using serverless vector search.
              </p>
            </div>

            {/* Perfectly Left-Aligned 2x2 Suggestion Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto pt-2">
              {suggestionCards.map((card, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(card.query)}
                  className="p-4 bg-white hover:bg-slate-50/80 border border-slate-200/80 hover:border-slate-300 rounded-2xl shadow-sm transition-all group duration-200 flex items-start justify-between text-left w-full cursor-pointer"
                >
                  <div className="text-left flex-1 min-w-0 pr-2">
                    <h4 className="text-xs font-bold text-slate-900 group-hover:text-slate-950 transition-colors truncate">
                      {card.title}
                    </h4>
                    <p className="text-[11px] text-slate-500 font-normal mt-1 leading-snug text-left line-clamp-2">
                      {card.query}
                    </p>
                  </div>
                  <ArrowUpRight className="w-3.5 h-3.5 text-slate-300 group-hover:text-slate-700 transition-colors flex-shrink-0 mt-0.5" strokeWidth={1.5} />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            {messages.map((msg, idx) => {
              const isCurrentlyStreaming = isStreaming && idx === messages.length - 1 && msg.role === "assistant";
              return (
                <div key={msg.id} className="space-y-2">
                  {/* User Message */}
                  {msg.role === "user" ? (
                    <div className="max-w-[960px] mx-auto w-full flex justify-end">
                      <div className="bg-[#f4f4f4] text-[#0d0d0d] rounded-[20px] px-5 py-2.5 max-w-[70%] text-[15px] font-normal leading-relaxed">
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      </div>
                    </div>
                  ) : (
                    /* AI Streaming Text - Left Aligned (ChatGPT Style) - Moved further left */
                    <div className="max-w-[960px] mx-auto w-full">
                      <div className="py-3 text-[#0d0d0d] space-y-2">
                        <div className="prose prose-slate prose-base max-w-none prose-p:leading-relaxed prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-pre:rounded-xl text-[#0d0d0d] text-[15.5px]">
                          {!msg.content && isCurrentlyStreaming ? (
                            <div className="flex items-center gap-1.5 py-3">
                              <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.3s]" />
                              <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce [animation-delay:-0.15s]" />
                              <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" />
                            </div>
                          ) : (
                            <>
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {msg.content}
                              </ReactMarkdown>
                              {isCurrentlyStreaming && (
                                <span className="inline-block w-2 h-4 ml-1 bg-slate-900 animate-pulse align-middle" />
                              )}
                            </>
                          )}
                        </div>

                        {/* Action Bar Below AI Response */}
                        {!isCurrentlyStreaming && msg.content && (
                          <div className="flex items-center gap-2 pt-1 text-xs text-slate-400">
                            <button
                              onClick={() => copyToClipboard(msg.content, msg.id)}
                              className="flex items-center gap-1 px-2 py-1 hover:bg-slate-100 hover:text-slate-700 rounded-md transition-colors text-[11px] cursor-pointer"
                              title="Copy Answer"
                            >
                              {copiedId === msg.id ? (
                                <>
                                  <Check className="w-3 h-3 text-emerald-600" strokeWidth={2} />
                                  <span className="text-emerald-600 font-medium">Copied</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="w-3 h-3" strokeWidth={1.5} />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>

                            <button
                              onClick={() => handleSend(messages[idx - 1]?.content || "Regenerate answer")}
                              className="flex items-center gap-1 px-2 py-1 hover:bg-slate-100 hover:text-slate-700 rounded-md transition-colors text-[11px] cursor-pointer"
                              title="Regenerate Answer"
                            >
                              <RefreshCw className="w-3 h-3" strokeWidth={1.5} />
                              <span>Regenerate</span>
                            </button>
                          </div>
                        )}

                        {/* Phase A: Citation Source Badges */}
                        {!isCurrentlyStreaming && msg.sources && msg.sources.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-100 mt-2">
                            <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mr-0.5">Sources:</span>
                            {msg.sources.map((src, sIdx) => (
                              <span
                                key={sIdx}
                                className="inline-flex items-center gap-1 px-2 py-0.5 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-full text-[10px] font-medium transition-colors cursor-default"
                                title={`${src.name}${src.page != null ? `, page ${src.page + 1}` : ""}`}
                              >
                                <FileText className="w-2.5 h-2.5 text-slate-400" strokeWidth={1.5} />
                                <span>{src.name}</span>
                                {src.page != null && (
                                  <span className="text-slate-400">· p.{src.page + 1}</span>
                                )}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Centered Input Bar with Attachment Paperclip Button */}
      <div className="p-4 max-w-3xl mx-auto w-full sticky bottom-0 bg-white">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="relative flex items-center bg-slate-100/70 border border-slate-200 focus-within:border-slate-400 focus-within:bg-white rounded-2xl shadow-sm transition-all pl-2"
        >
          {onUploadSuccess && (
            <DocumentUpload onUploadSuccess={onUploadSuccess} variant="icon" />
          )}

          <button
            type="button"
            onClick={toggleListening}
            className={`p-2 rounded-xl transition-all mr-1 cursor-pointer flex-shrink-0 ${
              isListening 
                ? "text-rose-600 bg-rose-50 animate-pulse hover:bg-rose-100" 
                : "text-slate-400 hover:text-slate-700 hover:bg-slate-100"
            }`}
            title={isListening ? "Listening... Click to stop" : "Start voice input"}
          >
            <Mic className="w-4 h-4" strokeWidth={1.5} />
          </button>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              selectedDocuments.length > 0 
                ? `Ask a question about ${selectedDocuments.length} PDF file(s)...`
                : "Ask anything about your documents..."
            }
            className="w-full pl-2 pr-12 py-3.5 bg-transparent text-sm text-slate-900 placeholder-slate-400 focus:outline-none font-medium"
          />

          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className="absolute right-2.5 p-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl transition-all shadow-sm disabled:opacity-30"
          >
            <Send className="w-3.5 h-3.5" strokeWidth={1.5} />
          </button>
        </form>
      </div>

    </div>
  );
}
