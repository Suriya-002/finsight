"""
Sentiment analysis pipeline for financial documents.
Provides both LLM-based and rule-based sentiment scoring.
"""

import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Financial sentiment lexicon
BULLISH_TERMS = {
    "beat", "exceeded", "upgrade", "outperform", "growth", "profit", "surge",
    "rally", "breakout", "upside", "bullish", "strong", "record", "positive",
    "raised guidance", "dividend increase", "buyback", "acquisition",
}

BEARISH_TERMS = {
    "miss", "below", "downgrade", "underperform", "decline", "loss", "plunge",
    "selloff", "breakdown", "downside", "bearish", "weak", "warning", "negative",
    "lowered guidance", "dividend cut", "layoff", "investigation", "default",
}


class SentimentAnalyzer:
    """Rule-based sentiment analysis as a fast fallback when LLM is unavailable."""

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Compute sentiment score from text using financial lexicon.
        
        Returns:
            Dict with score (-1 to 1), label, and matched terms.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        bullish_matches = words & BULLISH_TERMS
        bearish_matches = words & BEARISH_TERMS

        # Also check multi-word phrases
        for phrase in BULLISH_TERMS:
            if " " in phrase and phrase in text_lower:
                bullish_matches.add(phrase)
        for phrase in BEARISH_TERMS:
            if " " in phrase and phrase in text_lower:
                bearish_matches.add(phrase)

        bull_count = len(bullish_matches)
        bear_count = len(bearish_matches)
        total = bull_count + bear_count

        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (bull_count - bear_count) / total
            if score > 0.2:
                label = "bullish"
            elif score < -0.2:
                label = "bearish"
            else:
                label = "neutral"

        return {
            "score": round(score, 3),
            "label": label,
            "bullish_terms": list(bullish_matches),
            "bearish_terms": list(bearish_matches),
        }
