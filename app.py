# Production-Worthy PDF RAG Chatbot
# Complete implementation covering all 5 levels

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import uuid
import asyncio
from pathlib import Path

# Core dependencies
import streamlit as st
import PyPDF2
import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message with metadata"""
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime
    sources: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

@dataclass 
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = None
    
class ConfigManager:
    """Manages environment configuration and API keys"""
    
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT') 
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'docuchat-index')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Validate required keys
        if not all([self.pinecone_api_key, self.gemini_api_key]):
            raise ValueError("Missing required API keys in environment variables")
    
    def get_pinecone_config(self) -> Dict[str, str]:
        return {
            'api_key': self.pinecone_api_key,
            'environment': self.pinecone_environment,
            'index_name': self.pinecone_index_name
        }

class PDFProcessor:
    """Handles PDF parsing and text extraction with metadata"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract text from PDF with page-level metadata"""
        chunks = []
        
        try:
            # Use LangChain's PyPDFLoader for better metadata handling
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            for page_num, page in enumerate(pages):
                # Split page content into chunks
                page_chunks = self.text_splitter.split_text(page.page_content)
                
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    if chunk_text.strip():  # Skip empty chunks
                        metadata = {
                            'document_name': Path(pdf_path).name,
                            'page_number': page_num + 1,
                            'chunk_index': chunk_idx,
                            'file_path': pdf_path,
                            'extraction_timestamp': datetime.now().isoformat(),
                            'chunk_size': len(chunk_text)
                        }
                        
                        # Try to extract section titles (simple heuristic)
                        lines = chunk_text.split('\n')
                        potential_title = lines[0].strip() if lines else ""
                        if len(potential_title) < 100 and potential_title.isupper():
                            metadata['section_title'] = potential_title
                        
                        chunks.append(DocumentChunk(
                            content=chunk_text.strip(),
                            metadata=metadata
                        ))
                        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
            
        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        return chunks

class EmbeddingManager:
    """Manages text embeddings using Google's embedding models"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        
        # Using Google's embedding model for consistency with Gemini
        self.embedding_model = "models/embedding-001"
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = []
            
            # Process in batches to handle API limits
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                batch_embeddings = []
                for text in batch:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    batch_embeddings.append(result['embedding'])
                
                embeddings.extend(batch_embeddings)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

class VectorDatabase:
    """Manages Pinecone vector database operations"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index_name = config.pinecone_index_name
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Google embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def upsert_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks with embeddings in vector database"""
        try:
            vectors = []
            for i, chunk in enumerate(chunks):
                vector_id = f"{chunk.metadata['document_name']}_{chunk.metadata['page_number']}_{i}"
                
                vectors.append({
                    'id': vector_id,
                    'values': chunk.embedding,
                    'metadata': {
                        **chunk.metadata,
                        'content': chunk.content[:1000]  # Truncate for metadata storage
                    }
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks with optional metadata filtering"""
        try:
            query_params = {
                'vector': query_embedding,
                'top_k': top_k,
                'include_metadata': True
            }
            
            if filters:
                query_params['filter'] = filters
            
            results = self.index.query(**query_params)
            
            return [{
                'content': match['metadata']['content'],
                'score': match['score'],
                'metadata': match['metadata']
            } for match in results['matches']]
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise

class LLMManager:
    """Manages Gemini LLM interactions"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]], 
                         chat_history: List[ChatMessage] = None) -> str:
        """Generate response using Gemini with context and chat history"""
        try:
            # Build context string
            context_str = "\n\n".join([
                f"[Source: {ctx['metadata']['document_name']}, Page: {ctx['metadata']['page_number']}]\n{ctx['content']}"
                for ctx in context
            ])
            
            # Build chat history string
            history_str = ""
            if chat_history:
                history_str = "\n".join([
                    f"{msg.role.title()}: {msg.content}"
                    for msg in chat_history[-6:]  # Last 6 messages for context
                ])
            
            # Construct full prompt
            full_prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

Context from documents:
{context_str}

Chat History:
{history_str}

Current Question: {prompt}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when possible
- Be concise but comprehensive
- Maintain conversation continuity when relevant

Answer:"""

            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise

class ConversationMemory:
    """Manages conversation history and memory"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[ChatMessage] = []
        self.session_id = str(uuid.uuid4())
    
    def add_message(self, content: str, role: str, sources: List[Dict] = None):
        """Add a message to conversation history"""
        message = ChatMessage(
            id=str(uuid.uuid4()),
            content=content,
            role=role,
            timestamp=datetime.now(),
            sources=sources
        )
        
        self.messages.append(message)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 6) -> List[ChatMessage]:
        """Get recent messages for context"""
        return self.messages[-count:] if self.messages else []
    
    def clear_history(self):
        """Clear conversation history"""
        self.messages = []
        self.session_id = str(uuid.uuid4())

class AgentTools:
    """Tools for the RAG agent"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def search_documents(self, query: str, document_filter: str = None) -> str:
        """Search documents with optional filtering"""
        try:
            filters = {}
            if document_filter:
                filters['document_name'] = document_filter
            
            results = self.rag_system.search(query, filters=filters)
            
            if not results:
                return "No relevant documents found for the query."
            
            # Format results
            formatted_results = []
            for result in results:
                source_info = f"[{result['metadata']['document_name']}, Page {result['metadata']['page_number']}]"
                formatted_results.append(f"{source_info}\n{result['content']}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def summarize_document(self, document_name: str) -> str:
        """Summarize a specific document"""
        try:
            # Get all chunks from the document
            filters = {'document_name': document_name}
            results = self.rag_system.vector_db.search(
                query_embedding=[0] * 768,  # Dummy embedding for filter-only search
                top_k=100,
                filters=filters
            )
            
            if not results:
                return f"Document '{document_name}' not found."
            
            # Combine content
            full_content = "\n".join([r['content'] for r in results])
            
            # Use LLM to summarize
            summary_prompt = f"Please provide a comprehensive summary of this document:\n\n{full_content[:4000]}"
            summary = self.rag_system.llm.generate_response(summary_prompt, [])
            
            return f"Summary of {document_name}:\n{summary}"
            
        except Exception as e:
            return f"Error summarizing document: {str(e)}"

class RAGChatbot:
    """Main RAG Chatbot system integrating all components"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager(self.config)
        self.vector_db = VectorDatabase(self.config)
        self.llm = LLMManager(self.config)
        self.memory = ConversationMemory()
        self.agent_tools = AgentTools(self)
        
        # Document storage
        self.processed_documents: Dict[str, List[DocumentChunk]] = {}
    
    def upload_pdf(self, pdf_path: str) -> bool:
        """Upload and process a PDF document"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text chunks
            chunks = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_manager.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store in vector database
            self.vector_db.upsert_chunks(chunks)
            
            # Store locally for reference
            doc_name = Path(pdf_path).name
            self.processed_documents[doc_name] = chunks
            
            logger.info(f"Successfully processed {doc_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            
            # Search vector database
            results = self.vector_db.search(query_embedding, top_k, filters)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []
    
    def chat(self, user_query: str, document_filter: str = None) -> Tuple[str, List[Dict]]:
        """Main chat function with memory support"""
        try:
            # Prepare filters
            filters = {}
            if document_filter:
                filters['document_name'] = document_filter
            
            # Search for relevant context
            context = self.search(user_query, filters=filters)
            
            # Get chat history for context
            chat_history = self.memory.get_recent_messages()
            
            # Generate response
            response = self.llm.generate_response(user_query, context, chat_history)
            
            # Add to memory
            self.memory.add_message(user_query, "user")
            self.memory.add_message(response, "assistant", sources=context)
            
            return response, context
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}", []
    
    def get_document_list(self) -> List[str]:
        """Get list of processed documents"""
        return list(self.processed_documents.keys())
    
    def clear_conversation(self):
        """Clear conversation memory"""
        self.memory.clear_history()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Production PDF RAG Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Production PDF RAG Chatbot")
    st.markdown("Advanced RAG system with conversational memory, metadata filtering, and agent capabilities")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbot()
            st.success("✅ Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {str(e)}")
            st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process PDF
                        success = chatbot.upload_pdf(temp_path)
                        
                        # Clean up
                        os.remove(temp_path)
                        
                        if success:
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error(f"❌ Error processing {uploaded_file.name}")
        
        # Document filter
        st.header("🔍 Search Filters")
        documents = chatbot.get_document_list()
        document_filter = st.selectbox(
            "Filter by document (optional)",
            ["All documents"] + documents
        )
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            chatbot.clear_conversation()
            st.success("Conversation cleared!")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    if hasattr(chatbot.memory, 'messages') and chatbot.memory.messages:
        for message in chatbot.memory.messages:
            with st.chat_message(message.role):
                st.write(message.content)
                
                # Show sources for assistant messages
                if message.role == "assistant" and message.sources:
                    with st.expander("📚 Sources"):
                        for source in message.sources:
                            st.write(f"**{source['metadata']['document_name']}** (Page {source['metadata']['page_number']})")
                            st.write(f"Relevance: {source['score']:.3f}")
                            st.write(source['content'][:200] + "...")
                            st.divider()
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                filter_doc = None if document_filter == "All documents" else document_filter
                response, sources = chatbot.chat(user_query, document_filter=filter_doc)
                
                st.write(response)
                
                # Show sources
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"**{source['metadata']['document_name']}** (Page {source['metadata']['page_number']})")
                            st.write(f"Relevance: {source['score']:.3f}")
                            st.write(source['content'][:200] + "...")
                            st.divider()

if __name__ == "__main__":
    main()