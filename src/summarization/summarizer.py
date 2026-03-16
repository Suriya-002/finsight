"""
LLM-powered document summarization with sentiment analysis.
Uses AWS Bedrock Claude for structured output generation.
"""

import json
from typing import Any

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSummarizer:
    """Summarizes financial documents using AWS Bedrock Claude."""

    def __init__(self, config: dict):
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=config.get("aws_region", "us-east-1"),
        )
        self.model_id = config.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def summarize(self, text: str, source: str = "", title: str = "") -> dict[str, Any]:
        """
        Generate structured summary with sentiment and entity extraction.
        
        Returns:
            Dict with summary, sentiment, key_entities, material_events, key_metrics
        """
        if not text or len(text.strip()) < 50:
            return self._empty_result()

        prompt = f"""You are a financial analyst. Analyze this document and return ONLY a JSON object with these fields:
- "summary": 2-3 sentence summary of the key points relevant to investors
- "sentiment": exactly one of "bullish", "bearish", or "neutral"
- "key_entities": list of company names or ticker symbols mentioned
- "material_events": list of material events (earnings, M&A, guidance changes, regulatory actions, executive changes)
- "key_metrics": dict of any specific numbers mentioned (revenue, growth rate, price targets, etc.)

Title: {title}
Source: {source}

Document:
{text[:4000]}

Return ONLY valid JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 600,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )

            result = json.loads(response["body"].read())
            content = result["content"][0]["text"].strip()

            # Clean potential markdown wrapping
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)

            # Validate required fields
            return {
                "summary": parsed.get("summary", ""),
                "sentiment": parsed.get("sentiment", "neutral") if parsed.get("sentiment") in ["bullish", "bearish", "neutral"] else "neutral",
                "key_entities": parsed.get("key_entities", []),
                "material_events": parsed.get("material_events", []),
                "key_metrics": parsed.get("key_metrics", {}),
            }

        except json.JSONDecodeError:
            logger.warning("json_parse_failed", title=title[:50])
            return {
                "summary": content if 'content' in dir() else "",
                "sentiment": "neutral",
                "key_entities": [],
                "material_events": [],
                "key_metrics": {},
            }

        except Exception as e:
            logger.error("summarization_failed", error=str(e), title=title[:50])
            return self._empty_result()

    def summarize_batch(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Summarize a batch of articles, enriching each with summary data."""
        results = []
        for i, article in enumerate(articles):
            summary_data = self.summarize(
                text=article.get("content", ""),
                source=article.get("source", ""),
                title=article.get("title", ""),
            )
            results.append({**article, **summary_data})

            if (i + 1) % 10 == 0:
                logger.info("batch_progress", completed=i + 1, total=len(articles))

        logger.info("batch_summarization_complete", total=len(results))
        return results

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "summary": "",
            "sentiment": "neutral",
            "key_entities": [],
            "material_events": [],
            "key_metrics": {},
        }
