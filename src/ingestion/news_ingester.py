"""Financial news ingestion from multiple API sources."""

import httpx
import feedparser
from datetime import datetime, timedelta
from typing import Any

from src.summarization.summarizer import DocumentSummarizer
from src.search.embeddings import EmbeddingGenerator


class NewsIngester:
    """Ingests financial news from multiple sources and processes them."""

    def __init__(self, config: dict):
        self.sources = config.get("news_sources", [])
        self.summarizer = DocumentSummarizer(config)
        self.embedder = EmbeddingGenerator(config)

    async def ingest_batch(self, lookback_hours: int = 24) -> list[dict[str, Any]]:
        """Ingest news articles from all configured sources."""
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        for source in self.sources:
            if source["type"] == "rss":
                articles.extend(self._parse_rss(source["url"], cutoff))
            elif source["type"] == "api":
                articles.extend(await self._fetch_api(source, cutoff))

        # Process each article
        processed = []
        for article in articles:
            summary = self.summarizer.summarize(article["content"])
            embedding = self.embedder.generate(article["content"])
            processed.append({
                **article,
                "summary": summary["summary"],
                "sentiment": summary["sentiment"],
                "embedding": embedding,
                "processed_at": datetime.utcnow().isoformat(),
            })

        return processed

    def _parse_rss(self, url: str, cutoff: datetime) -> list[dict[str, Any]]:
        """Parse RSS feed for recent articles."""
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.utcnow()
            if pub_date >= cutoff:
                articles.append({
                    "title": entry.get("title", ""),
                    "content": entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "source": feed.feed.get("title", "RSS"),
                    "published_at": pub_date.isoformat(),
                })
        return articles

    async def _fetch_api(self, source: dict, cutoff: datetime) -> list[dict[str, Any]]:
        """Fetch articles from a news API."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                source["url"],
                params={"from": cutoff.isoformat(), "apiKey": source.get("api_key", "")},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": a.get("title", ""),
                    "content": a.get("description", ""),
                    "url": a.get("url", ""),
                    "source": source.get("name", "API"),
                    "published_at": a.get("publishedAt", ""),
                }
                for a in data.get("articles", [])
            ]
