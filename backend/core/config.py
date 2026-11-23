from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:3000"
    MAX_UPLOAD_MB: int = 25
    # MinIO Storage (Using as a placeholder for UploadThing/S3 in Cloud)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "documents"

    # Local LLM preference for machine-local inference
    USE_LOCAL_LLM: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:7b"

    # Cloud LLM & Embedding (Free Tier)
    GROQ_API_KEY: str = "" # Set this for $0 Llama 3!
    HF_TOKEN: str = "" # Set this for HuggingFace embeddings
    MAIN_LLM_MODEL: str = "openai/gpt-oss-20b"

    # Vector DB (Qdrant Cloud)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = "" # Set for Qdrant Cloud!

    # Supabase (Phase B - Persistence & Auth)
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_ANON_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
