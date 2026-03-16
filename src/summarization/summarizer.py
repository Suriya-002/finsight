"""LLM-powered document summarization with sentiment tagging."""

import json
import boto3
from typing import Any


class DocumentSummarizer:
    """Summarizes financial documents using AWS Bedrock Claude."""

    def __init__(self, config: dict):
        self.bedrock = boto3.client("bedrock-runtime", region_name=config.get("aws_region", "us-east-1"))
        self.model_id = config.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0")

    def summarize(self, text: str, max_tokens: int = 500) -> dict[str, Any]:
        """Generate summary and sentiment for a document."""
        prompt = f"""Analyze this financial document. Return JSON with:
- "summary": 2-3 sentence summary of key points
- "sentiment": one of ["bullish", "bearish", "neutral"]  
- "key_entities": list of companies/tickers mentioned
- "material_events": list of any material events (earnings, M&A, guidance)

Document:
{text[:4000]}"""

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }),
        )

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"summary": content, "sentiment": "neutral", "key_entities": [], "material_events": []}
