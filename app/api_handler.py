"""Shared API constants and request handlers."""

from __future__ import annotations

from app.analyzer import SentimentAnalyzer

SAMPLE_TEXTS = [
    "I absolutely love this product! Best purchase I've ever made.",
    "This is the worst experience I've had. Completely disappointed.",
    "The package arrived on Tuesday. It was a standard delivery.",
    "Not bad, but could definitely be improved in several areas.",
    "I'm thrilled with the customer support — they went above and beyond!",
]

_analyzer: SentimentAnalyzer | None = None


def get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def health_response() -> dict:
    return {"status": "ok", "service": "ai-sentiment-analyser"}


def samples_response() -> dict:
    return {"samples": SAMPLE_TEXTS}


def analyze_request(payload: dict) -> tuple[int, dict]:
    text = payload.get("text", "").strip()
    texts = payload.get("texts", [])

    try:
        analyzer = get_analyzer()
        if texts:
            results = [r.to_dict() for r in analyzer.analyze_batch(texts)]
            return 200, {"results": results}
        if text:
            return 200, analyzer.analyze(text).to_dict()
        return 400, {"error": "Provide 'text' or 'texts' in request body"}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": f"Analysis failed: {exc}"}
