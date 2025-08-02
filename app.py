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
import re
import unicodedata

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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads variables from .env file
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_id(text: str) -> str:
    """Convert text to ASCII-safe string for Pinecone IDs"""
    # Remove or replace non-ASCII characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace spaces and special characters with underscores
    text = re.sub(r'[^a-zA-Z0-9._-]', '_', text)
    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text

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
        # Try to get from Streamlit secrets first, then environment variables
        self.pinecone_api_key = None
        self.pinecone_environment = None
        self.pinecone_index_name = None
        self.gemini_api_key = None
        
        # Try Streamlit secrets first (for deployed apps)
        try:
            self.pinecone_api_key = st.secrets.get('PINECONE_API_KEY')
            self.pinecone_environment = st.secrets.get('PINECONE_ENVIRONMENT')
            self.pinecone_index_name = st.secrets.get('PINECONE_INDEX_NAME', 'docuchat-index')
            self.gemini_api_key = st.secrets.get('GEMINI_API_KEY')
        except:
            pass
        
        # Fallback to environment variables (for local development)
        if not self.pinecone_api_key:
            self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not self.pinecone_environment:
            self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        if not self.pinecone_index_name:
            self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'docuchat-index')
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Validate required keys
        if not all([self.pinecone_api_key, self.gemini_api_key]):
            missing_keys = []
            if not self.pinecone_api_key:
                missing_keys.append("PINECONE_API_KEY")
            if not self.gemini_api_key:
                missing_keys.append("GEMINI_API_KEY")
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        logger.info("Configuration loaded successfully")
    
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
                        if len(potential_title) < 100 and potential_title.upper() == potential_title:
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
            
            # Process in smaller batches to handle API limits
            batch_size = 10  # Reduced batch size for better reliability
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                batch_embeddings = []
                for text in batch:
                    # Truncate text if too long
                    if len(text) > 8000:
                        text = text[:8000]
                    
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
            # Truncate query if too long
            if len(query) > 8000:
                query = query[:8000]
                
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
                
                # Wait for index to be ready
                import time
                time.sleep(10)
            
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
                # Create ASCII-safe vector ID
                doc_name = sanitize_id(chunk.metadata['document_name'])
                vector_id = f"{doc_name}_{chunk.metadata['page_number']}_{i}"
                
                # Ensure metadata values are JSON serializable and not too large
                safe_metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        if isinstance(value, str) and len(value) > 1000:
                            safe_metadata[key] = value[:1000] + "..."
                        else:
                            safe_metadata[key] = value
                    else:
                        safe_metadata[key] = str(value)
                
                # Add truncated content to metadata
                safe_metadata['content'] = chunk.content[:1000] if len(chunk.content) > 1000 else chunk.content
                
                vectors.append({
                    'id': vector_id,
                    'values': chunk.embedding,
                    'metadata': safe_metadata
                })
            
            # Upsert in smaller batches for better reliability
            batch_size = 50  # Reduced batch size
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            
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
                # Sanitize filter values
                safe_filters = {}
                for key, value in filters.items():
                    if isinstance(value, str):
                        safe_filters[key] = sanitize_id(value) if key == 'document_name' else value
                    else:
                        safe_filters[key] = value
                query_params['filter'] = safe_filters
            
            results = self.index.query(**query_params)
            
            return [{
                'content': match['metadata'].get('content', ''),
                'score': match['score'],
                'metadata': match['metadata']
            } for match in results['matches']]
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise
    
    def delete_vectors_by_document(self, document_name: str):
        """Delete all vectors for a specific document from Pinecone"""
        try:
            # Create the sanitized document name used in vector IDs
            sanitized_name = sanitize_id(document_name)
            
            # Query all vectors for this document
            # We'll use a metadata filter to find all vectors for this document
            results = self.index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                filter={'document_name': sanitized_name}
            )
            
            if results['matches']:
                # Extract vector IDs
                vector_ids = [match['id'] for match in results['matches']]
                
                # Delete vectors in batches
                batch_size = 100
                for i in range(0, len(vector_ids), batch_size):
                    batch_ids = vector_ids[i:i + batch_size]
                    self.index.delete(ids=batch_ids)
                
                logger.info(f"Deleted {len(vector_ids)} vectors for document {document_name}")
                return len(vector_ids)
            else:
                logger.info(f"No vectors found for document {document_name}")
                return 0
                
        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_name}: {str(e)}")
            raise
    
    def delete_all_vectors(self):
        """Delete all vectors from the Pinecone index"""
        try:
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            logger.info("Deleted all vectors from Pinecone index")
        except Exception as e:
            logger.error(f"Error deleting all vectors: {str(e)}")
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
                f"[Source: {ctx['metadata'].get('document_name', 'Unknown')}, Page: {ctx['metadata'].get('page_number', 'Unknown')}]\n{ctx['content']}"
                for ctx in context[:5]  # Limit context to prevent token overflow
            ])
            
            # Build chat history string
            history_str = ""
            if chat_history:
                history_str = "\n".join([
                    f"{msg.role.title()}: {msg.content[:500]}"  # Truncate long messages
                    for msg in chat_history[-4:]  # Reduced history for token management
                ])
            
            # Construct full prompt with token management
            full_prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

Context from documents:
{context_str[:4000]}

Chat History:
{history_str[:1000]}

Current Question: {prompt}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when possible (document name and page number)
- Be concise but comprehensive
- Maintain conversation continuity when relevant

Answer:"""

            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"

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
    
    def get_recent_messages(self, count: int = 4) -> List[ChatMessage]:  # Reduced for token management
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
                filters['document_name'] = sanitize_id(document_filter)
            
            results = self.rag_system.search(query, filters=filters)
            
            if not results:
                return "No relevant documents found for the query."
            
            # Format results
            formatted_results = []
            for result in results:
                source_info = f"[{result['metadata'].get('document_name', 'Unknown')}, Page {result['metadata'].get('page_number', 'Unknown')}]"
                formatted_results.append(f"{source_info}\n{result['content']}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def summarize_document(self, document_name: str) -> str:
        """Summarize a specific document"""
        try:
            # Get all chunks from the document
            filters = {'document_name': sanitize_id(document_name)}
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
        """Upload and process a PDF document with better error handling"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Check file exists and is readable
            if not os.path.exists(pdf_path):
                logger.error(f"File not found: {pdf_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(pdf_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                logger.error(f"File too large: {file_size / (1024*1024):.1f}MB")
                return False
            
            # Extract text chunks with timeout protection
            try:
                chunks = self.pdf_processor.extract_text_from_pdf(pdf_path)
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                return False
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path}")
                return False
            
            # Limit number of chunks to prevent overflow
            if len(chunks) > 1000:
                logger.warning(f"Too many chunks ({len(chunks)}), limiting to 1000")
                chunks = chunks[:1000]
            
            # Generate embeddings with retry logic
            texts = [chunk.content for chunk in chunks]
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    embeddings = self.embedding_manager.generate_embeddings(texts)
                    break
                except Exception as e:
                    logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error("All embedding generation attempts failed")
                        return False
                    import time
                    time.sleep(2)  # Wait before retry
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store in vector database with retry logic
            for attempt in range(max_retries):
                try:
                    self.vector_db.upsert_chunks(chunks)
                    break
                except Exception as e:
                    logger.warning(f"Vector database upsert attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error("All vector database upsert attempts failed")
                        return False
                    import time
                    time.sleep(2)  # Wait before retry
            
            # Store locally for reference
            doc_name = Path(pdf_path).name
            if doc_name.startswith("temp_"):
                doc_name = doc_name[5:]  # Remove "temp_" prefix
            self.processed_documents[doc_name] = chunks
            
            logger.info(f"Successfully processed {doc_name} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks using the vector database"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)
            
            # Search vector database using the vector_db instance
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
            if document_filter and document_filter != "All documents":
                filters['document_name'] = sanitize_id(document_filter)
            
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
            error_response = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            return error_response, []
    
    def get_document_list(self) -> List[str]:
        """Get list of processed documents"""
        return list(self.processed_documents.keys())
    
    def clear_conversation(self):
        """Clear conversation memory only (keep documents)"""
        self.memory.clear_history()
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in the database"""
        try:
            stats = self.vector_db.index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 0
    
    def delete_document(self, document_name: str) -> bool:
        """Delete a specific document from both local storage and vector database"""
        try:
            # Delete from vector database
            deleted_count = self.vector_db.delete_vectors_by_document(document_name)
            
            # Delete from local storage
            if document_name in self.processed_documents:
                del self.processed_documents[document_name]
            
            # Clear conversation memory to avoid referencing deleted documents
            self.memory.clear_history()
            
            logger.info(f"Successfully deleted document {document_name} ({deleted_count} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_name}: {str(e)}")
            return False
    
    def delete_all_documents(self) -> bool:
        """Delete all documents from both local storage and vector database"""
        try:
            # Delete all vectors from database
            self.vector_db.delete_all_vectors()
            
            # Clear local storage
            self.processed_documents.clear()
            
            # Clear conversation memory
            self.memory.clear_history()
            
            logger.info("Successfully deleted all documents and cleared memory")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all documents: {str(e)}")
            return False

def process_single_file(chatbot, uploaded_file, status_text=None):
    """Process a single uploaded file with detailed progress tracking"""
    temp_path = None
    try:
        # Check file size
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            if status_text:
                status_text.error(f"❌ File too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
            else:
                st.error(f"❌ File too large: {file_size / (1024*1024):.1f}MB (max 10MB)")
            return False
        
        # Save uploaded file temporarily with progress
        temp_path = f"temp_{uploaded_file.name}"
        if status_text:
            status_text.text(f"💾 Saving {uploaded_file.name}...")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process PDF with progress updates
        if status_text:
            status_text.text(f"📖 Extracting text from {uploaded_file.name}...")
        
        # Extract text chunks
        chunks = chatbot.pdf_processor.extract_text_from_pdf(temp_path)
        
        if not chunks:
            if status_text:
                status_text.error(f"❌ No text found in {uploaded_file.name}")
            else:
                st.error(f"❌ No text found in {uploaded_file.name}")
            return False
        
        if status_text:
            status_text.text(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in smaller batches with progress
        texts = [chunk.content for chunk in chunks]
        embeddings = []
        
        batch_size = 5  # Smaller batch size for better progress tracking
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if status_text:
                status_text.text(f"🧠 Processing embeddings {i+1}-{min(i+batch_size, len(texts))} of {len(texts)}...")
            
            batch_embeddings = chatbot.embedding_manager.generate_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        if status_text:
            status_text.text(f"💾 Storing in vector database...")
        
        # Store in vector database
        chatbot.vector_db.upsert_chunks(chunks)
        
        # Store locally for reference
        doc_name = Path(temp_path).name.replace("temp_", "")
        chatbot.processed_documents[doc_name] = chunks
        
        if status_text:
            status_text.success(f"✅ Successfully processed {uploaded_file.name}")
        else:
            st.success(f"✅ {uploaded_file.name} processed successfully!")
        
        return True
        
    except Exception as e:
        error_msg = f"❌ Error processing {uploaded_file.name}: {str(e)}"
        logger.error(error_msg)
        
        if status_text:
            status_text.error(error_msg)
        else:
            st.error(error_msg)
        
        return False
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="DocuAI",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("🤖 DocuAI")
    st.markdown("Advanced RAG system with conversational memory, metadata filtering, and agent capabilities")
    
    # Check for required environment variables
    required_vars = ['PINECONE_API_KEY', 'GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        # Check both Streamlit secrets and environment variables
        value = None
        try:
            value = st.secrets.get(var)
        except:
            pass
        
        if not value:
            value = os.getenv(var)
        
        if not value:
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"❌ Missing required API keys: {', '.join(missing_vars)}")
        
        # Show current environment info for debugging
        st.info("🔍 **Debugging Info:**")
        st.write(f"- Current working directory: `{os.getcwd()}`")
        st.write(f"- Looking for `.env` file: `{os.path.exists('.env')}`")
        
        st.markdown("""
        ### How to add your API keys:
        
        **For Local Development:**
        1. Create a `.env` file in your project root directory
        2. Add your keys:
        ```env
        PINECONE_API_KEY=your_pinecone_api_key
        GEMINI_API_KEY=your_gemini_api_key
        PINECONE_INDEX_NAME=docuchat-index
        ```
        3. Make sure `python-dotenv` is installed: `pip install python-dotenv`
        
        **Alternative - Set environment variables directly:**
        ```bash
        export PINECONE_API_KEY=your_pinecone_api_key
        export GEMINI_API_KEY=your_gemini_api_key
        ```
        
        **For Streamlit Cloud:**
        1. Go to your app settings → Secrets tab
        2. Add your keys in TOML format:
        ```toml
        PINECONE_API_KEY = "your_pinecone_api_key"
        GEMINI_API_KEY = "your_gemini_api_key"
        PINECONE_INDEX_NAME = "docuchat-index"
        ```
        
        ### Get Your API Keys:
        - **Pinecone**: [app.pinecone.io](https://app.pinecone.io/) → API Keys
        - **Google Gemini**: [makersuite.google.com](https://makersuite.google.com/app/apikey)
        """)
        st.stop()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = RAGChatbot()
                st.success("✅ Chatbot initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing chatbot: {str(e)}")
            st.markdown("Please check your API keys and try again.")
            st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze (max 10MB each)"
        )
        
        # Initialize session state for processing status
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
        
        if uploaded_files:
            # Process all files button
            if len(uploaded_files) > 1:
                if st.button("🚀 Process All Files", type="primary"):
                    st.session_state.process_all = True
                st.divider()
            
            # Handle batch processing
            if st.session_state.get('process_all', False):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    if uploaded_file.name not in chatbot.get_document_list():
                        status_text.text(f"Processing {uploaded_file.name}...")
                        success = process_single_file(chatbot, uploaded_file, status_text)
                        
                        if success:
                            st.session_state.processing_status[uploaded_file.name] = "success"
                        else:
                            st.session_state.processing_status[uploaded_file.name] = "error"
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
                st.session_state.process_all = False
                st.rerun()
            
            # Individual file processing
            else:
                for i, uploaded_file in enumerate(uploaded_files):
                    # Create a container for each file
                    with st.container():
                        # File info (name and size)
                        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                        if file_size_mb > 10:
                            st.error(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f}MB) - File too large!")
                        else:
                            st.write(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f}MB)")
                        
                        # Status and action buttons in columns below filename
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Show status
                            if uploaded_file.name in chatbot.get_document_list():
                                st.success("✅ Ready for questions")
                            elif uploaded_file.name in st.session_state.processing_status:
                                if st.session_state.processing_status[uploaded_file.name] == "success":
                                    st.success("✅ Complete")
                                else:
                                    st.error("❌ Failed")
                            else:
                                st.info("⏳ Waiting")
                        
                        with col2:
                            # Process button with unique key
                            file_key = f"file_{i}_{hash(uploaded_file.name) % 10000}"
                            
                            if uploaded_file.name not in chatbot.get_document_list():
                                if file_size_mb <= 10:  # Only show button for valid files
                                    if st.button("🚀 Go", key=f"process_{file_key}", type="primary"):
                                        success = process_single_file(chatbot, uploaded_file)
                                        if success:
                                            st.session_state.processing_status[uploaded_file.name] = "success"
                                            st.rerun()
                                        else:
                                            st.session_state.processing_status[uploaded_file.name] = "error"
                                else:
                                    st.button("❌ Big", disabled=True, key=f"big_{file_key}")
                            else:
                                st.button("✅ OK", disabled=True, key=f"done_{file_key}")
                        
                        # Add separator between files
                        st.divider()
        
        # Document filter
        st.header("🔍 Search Filters")
        documents = chatbot.get_document_list()
        if documents:
            document_filter = st.selectbox(
                "Filter by document (optional)",
                ["All documents"] + documents,
                help="Restrict search to a specific document"
            )
        else:
            document_filter = "All documents"
            st.info("No documents uploaded yet")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", help="Clear chat history"):
            chatbot.clear_conversation()
            st.success("Conversation cleared!")
            st.rerun()
        
        # Document stats and management
        if documents:
            st.header("📊 Document Stats")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", len(documents))
            with col2:
                total_chunks = sum(len(chunks) for chunks in chatbot.processed_documents.values())
                st.metric("Text Chunks", total_chunks)
            with col3:
                vector_count = chatbot.get_vector_count()
                st.metric("Vectors in DB", vector_count)
            
            # Document list with remove option - vertical layout
            st.subheader("📋 Processed Documents")
            for i, doc in enumerate(documents):
                with st.container():
                    st.write(f"📄 **{doc}**")
                    
                    # Document details and remove button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        chunks_count = len(chatbot.processed_documents.get(doc, []))
                        st.caption(f"Contains {chunks_count} text chunks")
                    with col2:
                        # Unique key for remove button
                        doc_key = f"doc_{i}_{hash(doc) % 10000}"
                        if st.button("🗑️", key=f"rm_{doc_key}", help=f"Completely delete {doc}"):
                            with st.spinner(f"Deleting {doc} from database..."):
                                success = chatbot.delete_document(doc)
                                if success:
                                    st.success(f"✅ Completely deleted {doc}")
                                else:
                                    st.error(f"❌ Error deleting {doc}")
                                st.rerun()
                    
                    st.divider()
        
        # Clear all documents and memory management
        if documents:
            st.subheader("🗑️ Memory Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧹 Clear Chat Only", type="secondary", key="clear_chat_only", 
                           help="Clear conversation history but keep documents"):
                    chatbot.clear_conversation()
                    st.success("Chat history cleared!")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete All Data", type="secondary", key="delete_all_data", 
                           help="Delete ALL documents from database and memory"):
                    # Add confirmation
                    if 'confirm_delete_all' not in st.session_state:
                        st.session_state.confirm_delete_all = False
                    
                    if not st.session_state.confirm_delete_all:
                        st.session_state.confirm_delete_all = True
                        st.warning("⚠️ Click again to confirm deletion of ALL data!")
                        st.rerun()
                    else:
                        with st.spinner("Deleting all data from database..."):
                            success = chatbot.delete_all_documents()
                            if success:
                                st.success("✅ All data completely deleted!")
                                st.session_state.confirm_delete_all = False
                            else:
                                st.error("❌ Error deleting data")
                            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    if hasattr(chatbot.memory, 'messages') and chatbot.memory.messages:
        for message in chatbot.memory.messages:
            with st.chat_message(message.role):
                st.write(message.content)
                
                # Show sources for assistant messages
                if message.role == "assistant" and message.sources:
                    with st.expander("📄 Sources", expanded=False):
                        for i, source in enumerate(message.sources[:3]):  # Limit sources shown
                            st.write(f"**{source['metadata'].get('document_name', 'Unknown')}** (Page {source['metadata'].get('page_number', 'Unknown')})")
                            st.write(f"Relevance: {source['score']:.3f}")
                            st.write(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                            if i < len(message.sources) - 1:
                                st.divider()
    else:
        st.info("👋 Welcome! Upload some PDF documents and start asking questions about them.")
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        if not documents:
            st.warning("Please upload and process some PDF documents first!")
            st.stop()
        
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
                    with st.expander("📄 Sources", expanded=False):
                        for i, source in enumerate(sources[:3]):  # Limit sources shown
                            st.write(f"**{source['metadata'].get('document_name', 'Unknown')}** (Page {source['metadata'].get('page_number', 'Unknown')})")
                            st.write(f"Relevance: {source['score']:.3f}")
                            st.write(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                            if i < len(sources) - 1:
                                st.divider()

if __name__ == "__main__":
    main()