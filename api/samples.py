"""Vercel serverless handler for sample texts."""

import json
from http.server import BaseHTTPRequestHandler

from app.api_handler import samples_response


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps(samples_response()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
