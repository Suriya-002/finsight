"""Pinecone vector store operations."""

from typing import Any
from pinecone import Pinecone
from src.search.embeddings import EmbeddingGenerator


class VectorStore:
    def __init__(self, config: dict):
        self.pc = Pinecone(api_key=config.get("pinecone_api_key", ""))
        self.index = self.pc.Index(config.get("pinecone_index", "finsight"))
        self.embedder = EmbeddingGenerator(config)

    def upsert(self, doc_id: str, content: str, metadata: dict) -> None:
        embedding = self.embedder.generate(content)
        self.index.upsert(vectors=[{"id": doc_id, "values": embedding, "metadata": metadata}])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        embedding = self.embedder.generate(query)
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [
            {
                "content": m.metadata.get("content", ""),
                "source": m.metadata.get("source", ""),
                "date": m.metadata.get("date", ""),
                "url": m.metadata.get("url", ""),
                "score": m.score,
            }
            for m in results.matches
        ]
