import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from core.config import settings

# If no HF Token is provided, fallback to standard model, otherwise use HuggingFace API Embeddings
if settings.HF_TOKEN:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=settings.HF_TOKEN
    )
else:
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")

def process_document_task(document_id: str, file_uri: str, session_id: str = "default_session"):
    """
    Background task to process a PDF and embed it into Qdrant.
    No longer uses Celery! Built for Serverless environments using BackgroundTasks.
    """
    print(f"Starting background processing for document {document_id} (Session: {session_id}, URI: {file_uri})")
    
    # Download from MinIO
    local_path = f"/tmp/{document_id}.pdf"
    
    try:
        from services.document_service import document_service
        if document_service.use_s3:
            if f"s3://{document_service.bucket}/" in file_uri:
                object_name = file_uri.replace(f"s3://{document_service.bucket}/", "")
            else:
                object_name = file_uri.split("/")[-1] if "/" in file_uri else file_uri
                
            document_service.s3_client.download_file(document_service.bucket, object_name, local_path)
        else:
            local_path = file_uri
        
        # 2. Extract Text
        loader = PyPDFLoader(local_path)
        docs = loader.load()
        
        # 3. Chunk Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # 4. Generate Embeddings & Upsert to Qdrant
        if settings.QDRANT_API_KEY:
            qdrant_client = QdrantClient(
                url=settings.QDRANT_HOST,
                port=443,
                api_key=settings.QDRANT_API_KEY,
                https=True
            )
        else:
            qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
        vector_size = 384 if settings.HF_TOKEN else 768
        
        try:
            qdrant_client.get_collection("docuai_docs")
        except Exception:
            qdrant_client.create_collection(
                collection_name="docuai_docs",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

        try:
            from qdrant_client.http.models import PayloadSchemaType
            qdrant_client.create_payload_index(
                collection_name="docuai_docs",
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name="docuai_docs",
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass
            
        points = []
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk.page_content, 
                        "document_id": document_id, 
                        "session_id": session_id,
                        "page": chunk.metadata.get("page", 0)
                    }
                )
            )
            
        qdrant_client.upsert(
            collection_name="docuai_docs",
            points=points
        )
        print(f"Successfully upserted {len(points)} vectors to Qdrant for {document_id} (Session: {session_id})")
        
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return {"status": "success", "document_id": document_id, "session_id": session_id}
        
    except Exception as e:
        print(f"Failed to process document {document_id}: {str(e)}")
        raise e
