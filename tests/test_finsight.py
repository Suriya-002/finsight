"""Tests for FinSight pipeline components."""

import pytest
from src.summarization.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer:
    def test_bullish_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Revenue beat expectations with record growth and strong margins")
        assert result["label"] == "bullish"
        assert result["score"] > 0

    def test_bearish_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Company issued a warning as profits decline amid layoff announcements")
        assert result["label"] == "bearish"
        assert result["score"] < 0

    def test_neutral_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The company held its annual meeting on Tuesday in New York")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_empty_text(self):
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("")
        assert result["label"] == "neutral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
