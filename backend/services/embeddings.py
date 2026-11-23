import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


class LocalEmbeddings(Embeddings):
    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), threads=1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return LocalEmbeddings()
