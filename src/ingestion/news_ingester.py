"""
Financial news ingestion from multiple sources.
Supports RSS feeds, REST APIs, and SEC EDGAR with rate limiting and error handling.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsIngester:
    """Ingests financial news from multiple configurable sources."""

    def __init__(self, config: dict):
        self.sources = config.get("news_sources", [])
        self._rate_limit_delay = 1.0  # seconds between API calls

    async def ingest_batch(self, lookback_hours: int = 24) -> list[dict[str, Any]]:
        """
        Ingest news articles from all configured sources.
        
        Returns list of article dicts with title, content, source, url, published_at.
        """
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        all_articles = []

        for source in self.sources:
            try:
                if source["type"] == "rss":
                    articles = self._parse_rss(source, cutoff)
                elif source["type"] == "api":
                    articles = await self._fetch_api(source, cutoff)
                else:
                    logger.warning("unknown_source_type", source=source.get("name"), type=source["type"])
                    continue

                all_articles.extend(articles)
                logger.info("source_ingested", source=source.get("name"), articles=len(articles))

                await asyncio.sleep(self._rate_limit_delay)

            except Exception as e:
                logger.error("source_ingestion_failed", source=source.get("name"), error=str(e))

        logger.info("batch_ingestion_complete", total_articles=len(all_articles), sources=len(self.sources))
        return all_articles

    def _parse_rss(self, source: dict, cutoff: datetime) -> list[dict[str, Any]]:
        """Parse RSS feed for recent articles with robust date handling."""
        feed = feedparser.parse(source["url"])
        articles = []

        for entry in feed.entries:
            # Parse publication date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except (TypeError, ValueError):
                    pub_date = datetime.utcnow()
            else:
                pub_date = datetime.utcnow()

            if pub_date < cutoff:
                continue

            content = ""
            if hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "content") and entry.content:
                content = entry.content[0].get("value", "")

            # Strip HTML tags (basic)
            import re
            content = re.sub(r"<[^>]+>", "", content).strip()

            articles.append({
                "title": entry.get("title", "").strip(),
                "content": content[:5000],  # Cap content length
                "url": entry.get("link", ""),
                "source": source.get("name", feed.feed.get("title", "RSS")),
                "published_at": pub_date.isoformat(),
                "ingested_at": datetime.utcnow().isoformat(),
            })

        return articles

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_api(self, source: dict, cutoff: datetime) -> list[dict[str, Any]]:
        """Fetch articles from a news REST API with retry logic."""
        async with httpx.AsyncClient(timeout=30) as client:
            params = {
                "from": cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
                "sortBy": "publishedAt",
                "pageSize": 100,
            }
            if source.get("api_key"):
                params["apiKey"] = source["api_key"]

            resp = await client.get(source["url"], params=params)
            resp.raise_for_status()
            data = resp.json()

            return [
                {
                    "title": a.get("title", "").strip(),
                    "content": (a.get("description", "") or a.get("content", ""))[:5000],
                    "url": a.get("url", ""),
                    "source": source.get("name", "API"),
                    "published_at": a.get("publishedAt", datetime.utcnow().isoformat()),
                    "ingested_at": datetime.utcnow().isoformat(),
                }
                for a in data.get("articles", [])
                if a.get("title")
            ]
