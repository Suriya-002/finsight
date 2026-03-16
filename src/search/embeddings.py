"""Document embedding generation for vector search."""

import boto3
import json
from typing import Any


class EmbeddingGenerator:
    def __init__(self, config: dict):
        self.bedrock = boto3.client("bedrock-runtime", region_name=config.get("aws_region", "us-east-1"))

    def generate(self, text: str) -> list[float]:
        """Generate embedding vector for a text document."""
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text[:8000]}),
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
