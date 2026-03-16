"""RAG-based research assistant for querying ingested financial documents."""

import json
from typing import Any

import boto3
from src.search.vector_store import VectorStore


class RAGAssistant:
    """Natural language research assistant with source attribution."""

    def __init__(self, config: dict):
        self.vector_store = VectorStore(config)
        self.bedrock = boto3.client("bedrock-runtime", region_name=config.get("aws_region", "us-east-1"))
        self.model_id = config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Answer a research question using retrieved documents."""
        # Retrieve relevant documents
        docs = self.vector_store.search(question, top_k=top_k)

        # Build context
        context = "\n\n---\n\n".join(
            f"[Source: {d['source']} | Date: {d['date']}]\n{d['content']}"
            for d in docs
        )

        prompt = f"""You are a financial research assistant. Answer the question using ONLY 
the provided sources. Cite sources by [Source: name | Date: date].

Sources:
{context}

Question: {question}

Provide a clear, concise answer with source citations."""

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }),
        )

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]

        return {
            "answer": answer,
            "sources": [{"source": d["source"], "date": d["date"], "url": d.get("url", "")} for d in docs],
            "num_sources": len(docs),
        }
