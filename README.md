# FinSight — AI-Powered Financial News Monitor & Summarizer

An automated financial intelligence pipeline that ingests real-time news from 15+ sources, generates LLM-powered summaries with sentiment analysis, and delivers daily briefing reports to portfolio managers via a RAG-based research assistant.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![AWS](https://img.shields.io/badge/AWS-Bedrock%20%7C%20Lambda%20%7C%20PostgreSQL-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Semantic%20Search-purple)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Data Ingestion Layer                       │
│  SEC EDGAR  │  News APIs  │  Earnings Transcripts  │  RSS Feeds │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    AWS Lambda Triggers
                           │
              ┌────────────▼────────────┐
              │   LLM Summarization     │
              │   + Sentiment Tagging   │
              │   (AWS Bedrock Claude)  │
              └────────────┬────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      PostgreSQL      Pinecone       Excel Reports
      (Metadata)    (Embeddings)    (via openpyxl)
            │              │              │
            └──────────────┼──────────────┘
                           │
              ┌────────────▼────────────┐
              │   RAG Research Assistant │
              │   Natural Language Query │
              │   + Source Attribution   │
              └─────────────────────────┘
```

## Key Results

| Metric | Value |
|--------|-------|
| Data sources monitored | 15+ |
| Documents processed daily | 2,000+ |
| Query latency (RAG assistant) | < 2s |
| Alert detection accuracy | 94% |
| Daily briefing delivery | Automated (Excel + email) |

## Features

- **Multi-Source Ingestion** — SEC EDGAR, financial news APIs, earnings call transcripts, RSS feeds
- **LLM Summarization** — AWS Bedrock (Claude 3) generates concise summaries with key takeaways
- **Sentiment Analysis** — Document-level and entity-level sentiment scoring
- **Material Event Alerts** — Configurable thresholds for earnings surprises, M&A, guidance changes
- **RAG Research Assistant** — Natural language queries across all ingested documents with source attribution
- **Automated Briefings** — Daily Excel reports generated via openpyxl with portfolio-relevant highlights
- **Semantic Search** — Pinecone vector database for embedding-based document retrieval

## Tech Stack

`Python 3.11` · `AWS Bedrock (Claude 3)` · `AWS Lambda` · `Pinecone` · `PostgreSQL` · `FastAPI` · `Streamlit` · `openpyxl` · `BeautifulSoup` · `feedparser`

## Project Structure

```
finsight/
├── src/
│   ├── ingestion/
│   │   ├── sec_ingester.py        # SEC EDGAR filing ingestion
│   │   ├── news_ingester.py       # Financial news API ingestion
│   │   └── rss_ingester.py        # RSS feed monitoring
│   ├── summarization/
│   │   ├── summarizer.py          # LLM-powered summarization
│   │   └── sentiment.py           # Sentiment analysis pipeline
│   ├── search/
│   │   ├── embeddings.py          # Document embedding generation
│   │   ├── vector_store.py        # Pinecone operations
│   │   └── rag_assistant.py       # RAG query engine
│   └── alerts/
│       ├── detector.py            # Material event detection
│       └── report_generator.py    # Excel briefing generation
├── config/
│   └── sources.yaml               # Data source configurations
├── tests/
│   ├── test_ingestion.py
│   └── test_search.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
cp config/sources.yaml.example config/sources.yaml
# Configure API keys and database connections
python -m src.ingestion.news_ingester --backfill 7
python -m src.search.rag_assistant --interactive
```

## License

MIT
