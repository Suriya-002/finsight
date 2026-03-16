"""
RAG-based research assistant for querying ingested financial documents.
Supports natural language queries with source attribution and configurable retrieval.
"""

import json
from typing import Any

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from src.search.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGAssistant:
    """Natural language research assistant with source attribution."""

    def __init__(self, config: dict):
        self.vector_store = VectorStore(config)
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=config.get("aws_region", "us-east-1"),
        )
        self.model_id = config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, question: str, top_k: int = 5, min_score: float = 0.5) -> dict[str, Any]:
        """
        Answer a research question using retrieved documents.
        
        Args:
            question: Natural language research question
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info("rag_query", question=question[:100], top_k=top_k)

        # Retrieve relevant documents
        docs = self.vector_store.search(question, top_k=top_k)

        # Filter by minimum score
        docs = [d for d in docs if d.get("score", 0) >= min_score]

        if not docs:
            return {
                "answer": "I couldn't find any relevant documents to answer this question. Try rephrasing or broadening your query.",
                "sources": [],
                "num_sources": 0,
                "confidence": "low",
            }

        # Build context with source attribution markers
        context_parts = []
        for i, d in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}: {d.get('source', 'Unknown')} | {d.get('date', 'Unknown date')}]\n"
                f"{d.get('content', '')[:1500]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a financial research assistant at an investment firm. Answer the question 
using ONLY the provided sources. For each claim, cite the source as [Source N].
If the sources don't contain enough information, say so clearly.
Be concise and focus on actionable insights.

Sources:
{context}

Question: {question}

Provide a clear, concise answer with source citations."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2048,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )

            result = json.loads(response["body"].read())
            answer = result["content"][0]["text"]

            sources_used = [
                {
                    "source": d.get("source", ""),
                    "date": d.get("date", ""),
                    "url": d.get("url", ""),
                    "score": round(d.get("score", 0), 3),
                    "title": d.get("title", ""),
                }
                for d in docs
            ]

            logger.info("rag_query_complete", sources_used=len(sources_used))

            return {
                "answer": answer,
                "sources": sources_used,
                "num_sources": len(sources_used),
                "confidence": "high" if len(docs) >= 3 else "medium" if len(docs) >= 1 else "low",
            }

        except Exception as e:
            logger.error("rag_generation_failed", error=str(e))
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "confidence": "none",
            }

    def interactive_session(self):
        """Run an interactive query session in the terminal."""
        print("\n=== FinSight Research Assistant ===")
        print("Ask questions about ingested financial documents.")
        print("Type 'quit' to exit.\n")

        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue

            result = self.query(question)
            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                print(f"\nSources ({result['num_sources']}):")
                for s in result["sources"]:
                    print(f"  - {s['source']} ({s['date']}) [score: {s['score']}]")
            print()
