from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.config import settings
from langchain_qdrant import QdrantVectorStore

COLLECTION_NAME = "docuai_docs"

class RAGService:
    def __init__(self):
        # 1. Initialize Serverless LLM (Groq) or fallback to Local (Ollama)
        if settings.GROQ_API_KEY:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model_name=settings.MAIN_LLM_MODEL,
                temperature=0.3
            )
        else:
            from langchain_community.llms import Ollama
            self.llm = Ollama(
                base_url="http://localhost:11434",
                model="qwen2.5-coder:7b"
            )

        # 2. Initialize Serverless Embeddings (HuggingFace API) or fallback to Local
        if settings.HF_TOKEN:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=settings.HF_TOKEN
            )
        else:
            from langchain_community.embeddings import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434", 
                model="nomic-embed-text"
            )
        
        # 3. Initialize Cloud Vector Database (Qdrant Cloud) or fallback to Local
        if settings.QDRANT_API_KEY:
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_HOST,
                port=443,
                api_key=settings.QDRANT_API_KEY,
                https=True
            )
        else:
            self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
        # 4. Ensure collection exists in Qdrant Cloud/Local before initializing store
        try:
            self.qdrant_client.get_collection(COLLECTION_NAME)
        except Exception:
            from qdrant_client.http.models import VectorParams, Distance
            vector_size = 384 if settings.HF_TOKEN else 768
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        # 5. Create payload index for document_id and session_id (required by Qdrant for filtering)
        try:
            self.qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="session_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
            content_payload_key="text"
        )

    def _get_context(self, query: str, document_ids: list = None, session_id: str = "default_session") -> str:
        try:
            must_conditions = [
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            ]
            if document_ids and len(document_ids) > 0:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids)
                    )
                )
                
            filter_obj = models.Filter(must=must_conditions)
                
            docs = self.vector_store.similarity_search(query, k=8, filter=filter_obj)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"Vector search failed: {e}")
            # If payload index is missing, create it and retry once!
            if "Index required" in str(e):
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name="document_id",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    self.qdrant_client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name="session_id",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    docs = self.vector_store.similarity_search(query, k=8, filter=filter_obj)
                    return "\n\n---\n\n".join([doc.page_content for doc in docs])
                except Exception as retry_err:
                    print(f"Retry after index creation failed: {retry_err}")
            return ""
            
    async def stream_response(self, query: str, document_ids: list = None, session_id: str = "default_session"):
        context = self._get_context(query, document_ids, session_id=session_id)
        
        if context.strip():
            prompt = f"""You are DocuAI Enterprise, an elite document analysis AI.
Use the following retrieved context to answer the user's question comprehensively and accurately.
Provide detailed explanations, structure your answer well, and reference the provided context.
If the answer is not in the context, clearly state that the documents do not contain the answer.

Context:
{context}

User Question: {query}
Answer in a detailed, professional format:"""
        else:
            prompt = f"User Question: {query}\nAnswer comprehensively and professionally as an elite AI assistant:"

        try:
            async for chunk in self.llm.astream(prompt):
                # Groq returns AIMessage objects in astream, Ollama returns strings
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield chunk
        except Exception as e:
            yield f"Error generating response: {str(e)}"

rag_service = RAGService()
