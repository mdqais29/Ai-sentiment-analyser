"""Local development server for AI Sentiment Analyser."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.analyzer import SentimentAnalyzer

PUBLIC_DIR = Path(__file__).parent / "public"
analyzer = SentimentAnalyzer()

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath: Path) -> None:
        suffix = filepath.suffix.lower()
        content_type = MIME_TYPES.get(suffix, "application/octet-stream")
        body = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json(200, {"status": "ok", "service": "ai-sentiment-analyser"})
            return

        if path == "/api/samples":
            self._send_json(200, {"samples": SAMPLE_TEXTS})
            return

        if path in ("/", "/index.html"):
            return self._send_file(PUBLIC_DIR / "index.html")

        file_path = (PUBLIC_DIR / path.lstrip("/")).resolve()
        public_root = PUBLIC_DIR.resolve()
        if file_path.is_file() and public_root in file_path.parents:
            return self._send_file(file_path)

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")

        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        text = payload.get("text", "").strip()
        texts = payload.get("texts", [])

        try:
            if texts:
                results = [r.to_dict() for r in analyzer.analyze_batch(texts)]
                self._send_json(200, {"results": results})
            elif text:
                result = analyzer.analyze(text)
                self._send_json(200, result.to_dict())
            else:
                self._send_json(400, {"error": "Provide 'text' or 'texts' in request body"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": f"Analysis failed: {exc}"})


SAMPLE_TEXTS = [
    "I absolutely love this product! Best purchase I've ever made.",
    "This is the worst experience I've had. Completely disappointed.",
    "The package arrived on Tuesday. It was a standard delivery.",
    "Not bad, but could definitely be improved in several areas.",
    "I'm thrilled with the customer support — they went above and beyond!",
]


def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.allow_reuse_address = True
    print(f"AI Sentiment Analyser running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
