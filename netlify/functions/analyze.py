"""Netlify serverless function for sentiment analysis."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyzer import SentimentAnalyzer

_analyzer = None

SAMPLE_TEXTS = [
    "I absolutely love this product! Best purchase I've ever made.",
    "This is the worst experience I've had. Completely disappointed.",
    "The package arrived on Tuesday. It was a standard delivery.",
    "Not bad, but could definitely be improved in several areas.",
    "I'm thrilled with the customer support — they went above and beyond!",
]

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def handler(event, context):
    method = event.get("httpMethod", "GET")
    path = event.get("path", "")
    raw_path = event.get("rawPath", path)

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": HEADERS, "body": ""}

    if method == "GET":
        if raw_path.endswith("/health") or path.endswith("/health"):
            return {
                "statusCode": 200,
                "headers": HEADERS,
                "body": json.dumps({"status": "ok", "service": "ai-sentiment-analyser"}),
            }
        return {
            "statusCode": 200,
            "headers": HEADERS,
            "body": json.dumps({"samples": SAMPLE_TEXTS}),
        }

    if method == "POST":
        try:
            body = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": HEADERS,
                "body": json.dumps({"error": "Invalid JSON body"}),
            }

        text = body.get("text", "").strip()
        texts = body.get("texts", [])

        try:
            analyzer = get_analyzer()
            if texts:
                results = [r.to_dict() for r in analyzer.analyze_batch(texts)]
                return {
                    "statusCode": 200,
                    "headers": HEADERS,
                    "body": json.dumps({"results": results}),
                }
            if text:
                result = analyzer.analyze(text)
                return {
                    "statusCode": 200,
                    "headers": HEADERS,
                    "body": json.dumps(result.to_dict()),
                }
            return {
                "statusCode": 400,
                "headers": HEADERS,
                "body": json.dumps({"error": "Provide 'text' or 'texts' in request body"}),
            }
        except ValueError as exc:
            return {
                "statusCode": 400,
                "headers": HEADERS,
                "body": json.dumps({"error": str(exc)}),
            }
        except Exception as exc:
            return {
                "statusCode": 500,
                "headers": HEADERS,
                "body": json.dumps({"error": f"Analysis failed: {exc}"}),
            }

    return {
        "statusCode": 404,
        "headers": HEADERS,
        "body": json.dumps({"error": "Not found"}),
    }
