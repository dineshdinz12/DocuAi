"use client";

import { useState, useEffect } from "react";
import { DocumentUpload } from "@/components/DocumentUpload";
import { ChatInterface } from "@/components/ChatInterface";
import { AuthModal } from "@/components/AuthModal";
import { Sparkles, Settings, FileText, Trash2, ExternalLink, User, LogOut, LogIn } from "lucide-react";

import { getSessionId, getCurrentUser, clearUserSession, UserProfile } from "@/utils/session";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";

export interface Document {
  id: string;
  name: string;
  key: string;
  size: number;
  uploaded_at: string;
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  useEffect(() => {
    setUser(getCurrentUser());
  }, []);

  const fetchDocuments = async (retries = 3) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents`, {
        headers: { "x-session-id": getSessionId() }
      });
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

  useEffect(() => {
    fetchDocuments();
  }, []);

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

  const handleView = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/documents/${id}/url`, {
        headers: { "x-session-id": getSessionId() }
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.url) {
        window.open(data.url, "_blank");
      }
    } catch (err) {
      console.error("Failed to get URL", err);
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedDocIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="flex h-screen bg-white text-gray-900 font-sans">
      {/* Sidebar - ChatGPT Style */}
      <aside className="w-[300px] bg-[#f9f9f9] border-r border-gray-200 flex-col hidden md:flex h-full">
        <div className="p-5 border-b border-gray-200">
          <div className="flex items-center gap-2.5 font-bold text-xl mb-6">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-md shadow-indigo-500/20 text-white">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-900 bg-clip-text text-transparent">DocuAI</span>
          </div>
          
          <div className="mb-2">
            <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Memory Ingestion</p>
            <DocumentUpload onUploadSuccess={fetchDocuments} />
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Document Memory</p>
              <span className="text-[10px] text-gray-400 bg-gray-200 px-2 py-0.5 rounded-full">{documents.length}</span>
            </div>
            
            <div className="space-y-1">
              {documents.length === 0 ? (
                <p className="text-xs text-gray-400 text-center py-4 italic">No documents uploaded</p>
              ) : (
                documents.map(doc => (
                  <div 
                    key={doc.id}
                    onClick={() => toggleSelection(doc.id)}
                    className={`flex items-center gap-3 p-2.5 rounded-lg cursor-pointer text-sm transition-colors group border
                      ${selectedDocIds.includes(doc.id) 
                        ? 'bg-indigo-50 border-indigo-200 text-indigo-900' 
                        : 'bg-white border-transparent hover:border-gray-200 hover:bg-gray-50 text-gray-700'
                      }
                    `}
                  >
                    <div className="flex-1 flex items-center min-w-0 gap-2">
                      <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors
                        ${selectedDocIds.includes(doc.id) ? 'bg-indigo-600 border-indigo-600' : 'border-gray-300'}
                      `}>
                        {selectedDocIds.includes(doc.id) && <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                      </div>
                      <span className="truncate text-xs font-medium">{doc.name}</span>
                    </div>
                    
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={(e) => handleView(doc.id, e)} className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-indigo-600" title="View PDF">
                        <ExternalLink className="w-3.5 h-3.5" />
                      </button>
                      <button onClick={(e) => handleDelete(doc.id, e)} className="p-1 hover:bg-red-100 rounded text-gray-500 hover:text-red-600" title="Delete">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-200 bg-[#f9f9f9] space-y-2">
          {user ? (
            <div className="p-2.5 rounded-xl bg-white border border-gray-200 shadow-sm flex items-center justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-700 flex items-center justify-center font-bold text-xs flex-shrink-0">
                  {user.avatar_url ? (
                    <img src={user.avatar_url} alt={user.name} className="w-full h-full rounded-full object-cover" />
                  ) : (
                    user.name.charAt(0).toUpperCase()
                  )}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-bold text-gray-900 truncate">{user.name}</p>
                  <p className="text-[10px] text-gray-500 capitalize">{user.auth_type} Auth</p>
                </div>
              </div>
              <button
                onClick={() => {
                  clearUserSession();
                  setUser(null);
                  fetchDocuments();
                }}
                className="p-1.5 hover:bg-red-50 text-gray-400 hover:text-red-600 rounded-lg transition-colors"
                title="Sign Out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setIsAuthModalOpen(true)}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs rounded-xl transition-all shadow-sm shadow-indigo-200"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In / Sign Up</span>
            </button>
          )}

          <div className="flex items-center gap-3 p-2.5 rounded-lg hover:bg-gray-200 cursor-pointer text-sm text-gray-700 transition-colors">
            <Settings className="w-4 h-4 text-gray-500" />
            <span>Settings & Models</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative h-full bg-white">
        {/* Mobile header */}
        <header className="md:hidden flex items-center gap-2 p-4 border-b border-gray-200 bg-white sticky top-0 z-10">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center text-white">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <span className="font-extrabold text-lg bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">DocuAI</span>
        </header>
        
        <div className="flex-1 overflow-hidden relative">
          <ChatInterface selectedDocuments={documents.filter(d => selectedDocIds.includes(d.id))} />
        </div>
      </main>

      {/* Auth Modal Component */}
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
