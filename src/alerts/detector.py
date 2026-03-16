"""Material event detection with configurable alert thresholds."""

from typing import Any


class EventDetector:
    """Detects material events from summarized documents."""

    DEFAULT_THRESHOLDS = {
        "earnings_surprise": 0.05,
        "guidance_change": True,
        "ma_activity": True,
        "executive_change": True,
    }

    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def detect(self, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scan summaries for material events exceeding thresholds."""
        alerts = []
        for s in summaries:
            events = s.get("material_events", [])
            if events:
                alerts.append({
                    "source": s.get("source"),
                    "title": s.get("title"),
                    "events": events,
                    "sentiment": s.get("sentiment"),
                    "severity": "high" if s.get("sentiment") in ["bullish", "bearish"] else "medium",
                })
        return alerts
