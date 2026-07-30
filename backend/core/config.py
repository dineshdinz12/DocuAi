from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MinIO Storage (Using as a placeholder for UploadThing/S3 in Cloud)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "documents"

    # Cloud LLM & Embedding (Free Tier)
    GROQ_API_KEY: str = "" # Set this for $0 Llama 3!
    HF_TOKEN: str = "" # Set this for HuggingFace embeddings
    MAIN_LLM_MODEL: str = "llama-3.1-8b-instant"

    # Vector DB (Qdrant Cloud)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = "" # Set for Qdrant Cloud!

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
