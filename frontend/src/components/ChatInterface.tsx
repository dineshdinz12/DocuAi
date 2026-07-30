"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles, FileText, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Document } from "@/app/page";

import { getSessionId } from "@/utils/session";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ChatInterfaceProps {
  selectedDocuments: Document[];
}

export function ChatInterface({ selectedDocuments }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Dynamic context-aware suggestions
  const baseSuggestions = [
    "Give me a detailed summary of the selected documents.",
    "Extract all the specific technologies and tools mentioned.",
    "What are the main objectives discussed in these files?",
    "List the key projects or findings in a markdown table format."
  ];

  const filteredSuggestions = input.trim().length > 0
    ? baseSuggestions.filter(s => s.toLowerCase().includes(input.toLowerCase()) && s.toLowerCase() !== input.toLowerCase())
    : baseSuggestions;

  const suggestionCards = selectedDocuments.length > 0 
    ? [
        { title: `Summarize ${selectedDocuments[0].name.slice(0,10)}...`, desc: "Get a quick overview of this specific file" },
        { title: "Extract technologies", desc: "List all tools and languages" },
        { title: "Key achievements", desc: "Highlight major metrics and successes" },
        { title: "Projects overview", desc: "Format projects into a markdown table" }
      ]
    : [
        { title: "Upload a document first", desc: "Drag and drop a PDF into the sidebar" },
        { title: "General AI Chat", desc: "Ask me anything without document context" }
      ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async (query: string = input) => {
    if (!query.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
    setShowSuggestions(false);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const res = await fetch(`${baseUrl}/api/v1/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "x-session-id": getSessionId()
        },
        body: JSON.stringify({ 
          query,
          document_ids: selectedDocuments.map(d => d.id)
        }),
      });
      
      if (!res.body) throw new Error("No response body");
      
      setIsTyping(false); 
      
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botResponse = "";
      const messageId = (Date.now() + 1).toString();
      
      setMessages((prev) => [...prev, { id: messageId, role: "assistant", content: "" }]);
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        botResponse += chunk;
        
        setMessages((prev) => 
          prev.map((msg) => 
            msg.id === messageId ? { ...msg, content: botResponse } : msg
          )
        );
      }
    } catch (error) {
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Error: Could not connect to the backend server.",
      }]);
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto w-full scrollbar-thin scrollbar-thumb-gray-200">
        
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center px-4 pt-10 pb-32">
            <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center shadow-xl shadow-indigo-200 mb-8">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">What can I help you with?</h1>
            <p className="text-gray-500 mb-10 text-center max-w-md">
              Select documents from the sidebar to inject them into my memory, then ask me anything.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl w-full">
              {suggestionCards.map((card, idx) => (
                <div 
                  key={idx}
                  onClick={() => handleSend(card.title)}
                  className="p-4 border border-gray-200 rounded-xl hover:border-indigo-400 hover:shadow-md cursor-pointer transition-all bg-gray-50 hover:bg-white group"
                >
                  <p className="font-semibold text-gray-800 text-sm mb-1 group-hover:text-indigo-600">{card.title}</p>
                  <p className="text-xs text-gray-500">{card.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto py-12 px-4 sm:px-6 w-full flex flex-col gap-10 pb-40 pt-16">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-5 w-full ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                {msg.role === "assistant" && (
                  <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-0.5 text-white shadow-md">
                    <Sparkles className="w-4.5 h-4.5" />
                  </div>
                )}
                
                <div 
                  className={`flex flex-col max-w-[85%] ${
                    msg.role === "user" 
                      ? "bg-gray-100/80 rounded-3xl px-6 py-3.5 text-gray-800 shadow-sm" 
                      : "text-gray-800 pt-1"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <div className="prose prose-slate max-w-none prose-p:leading-relaxed prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:text-gray-800 prose-headings:font-semibold prose-a:text-indigo-600 prose-li:marker:text-gray-400">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content || "..."}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap text-[15px]">{msg.content}</div>
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex gap-5 w-full justify-start">
                <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0 mt-0.5 text-white shadow-md">
                  <Sparkles className="w-4.5 h-4.5" />
                </div>
                <div className="pt-3">
                  <div className="flex gap-1.5 items-center h-4">
                    <div className="w-2 h-2 rounded-full bg-indigo-400/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-2 h-2 rounded-full bg-indigo-400/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-2 h-2 rounded-full bg-indigo-400/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white/95 to-transparent pt-12 pb-6 px-4 pointer-events-none">
        <div className="max-w-3xl mx-auto w-full relative pointer-events-auto">
          
          {/* Active Memory Indicators */}
          {selectedDocuments.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3 px-2">
              <span className="text-xs font-semibold text-gray-500 uppercase flex items-center mr-1">Active Memory:</span>
              {selectedDocuments.map(doc => (
                <div key={doc.id} className="flex items-center gap-1.5 bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full text-xs font-medium border border-indigo-100 shadow-sm">
                  <FileText className="w-3 h-3" />
                  <span className="truncate max-w-[150px]">{doc.name}</span>
                </div>
              ))}
            </div>
          )}

          {/* Dynamic Suggestions (Autocomplete) */}
          {showSuggestions && filteredSuggestions.length > 0 && selectedDocuments.length > 0 && (
            <div className="absolute bottom-full left-0 mb-3 w-full px-2">
              <div className="flex flex-wrap gap-2">
                {filteredSuggestions.slice(0, 3).map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInput(suggestion);
                      handleSend(suggestion);
                    }}
                    className="flex items-center gap-1.5 bg-white border border-gray-200 shadow-sm rounded-full px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 hover:text-indigo-600 transition-colors"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span className="truncate max-w-[200px] sm:max-w-[300px]">{suggestion}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="relative flex items-end w-full border border-gray-200 bg-white/70 backdrop-blur-xl rounded-[2rem] shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden focus-within:ring-2 focus-within:ring-indigo-100 focus-within:border-indigo-400 focus-within:bg-white transition-all duration-300">
            <textarea
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (e.target.value.length > 0) setShowSuggestions(true);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={selectedDocuments.length > 0 ? "Ask anything about selected documents..." : "Ask a general question..."}
              className="w-full max-h-48 min-h-[60px] py-4 pl-6 pr-14 bg-transparent border-none focus:ring-0 resize-none text-[15px] text-gray-800 placeholder-gray-400 scrollbar-thin"
              rows={1}
            />
            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || isTyping}
              className="absolute right-2 bottom-2 p-3 rounded-full bg-indigo-600 text-white disabled:bg-gray-100 disabled:text-gray-300 transition-all duration-200 hover:bg-indigo-700 hover:scale-105 active:scale-95 disabled:hover:scale-100"
            >
              <Send className="w-4 h-4 translate-x-[-1px] translate-y-[1px]" />
            </button>
          </div>
          <div className="text-center mt-3">
            <span className="text-[11px] text-gray-400 font-medium">DocuAI can make mistakes. Verify critical info.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
