"""Vercel serverless handler for sentiment analysis."""

import json
from http.server import BaseHTTPRequestHandler

from app.analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

SAMPLE_TEXTS = [
    "I absolutely love this product! Best purchase I've ever made.",
    "This is the worst experience I've had. Completely disappointed.",
    "The package arrived on Tuesday. It was a standard delivery.",
    "Not bad, but could definitely be improved in several areas.",
    "I'm thrilled with the customer support — they went above and beyond!",
]


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path.endswith("/health"):
            self._send(200, {"status": "ok", "service": "ai-sentiment-analyser"})
            return

        if path.endswith("/samples"):
            self._send(200, {"samples": SAMPLE_TEXTS})
            return

        self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if not path.endswith("/analyze"):
            self._send(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")

        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send(400, {"error": "Invalid JSON body"})
            return

        text = payload.get("text", "").strip()
        texts = payload.get("texts", [])

        try:
            if texts:
                results = [r.to_dict() for r in analyzer.analyze_batch(texts)]
                self._send(200, {"results": results})
            elif text:
                result = analyzer.analyze(text)
                self._send(200, result.to_dict())
            else:
                self._send(400, {"error": "Provide 'text' or 'texts' in request body"})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": f"Analysis failed: {exc}"})
