"""Netlify serverless function for sentiment analysis."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api_handler import analyze_request, health_response, samples_response

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def handler(event, context):
    method = event.get("httpMethod", "GET")
    path = event.get("path", "")
    raw_path = event.get("rawPath", path)

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": HEADERS, "body": ""}

    if method == "GET":
        if raw_path.endswith("/health") or path.endswith("/health"):
            body = health_response()
        else:
            body = samples_response()
        return {
            "statusCode": 200,
            "headers": HEADERS,
            "body": json.dumps(body),
        }

    if method == "POST":
        try:
            payload = json.loads(event.get("body") or "{}")
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": HEADERS,
                "body": json.dumps({"error": "Invalid JSON body"}),
            }

        status, body = analyze_request(payload)
        return {
            "statusCode": status,
            "headers": HEADERS,
            "body": json.dumps(body),
        }

    return {
        "statusCode": 404,
        "headers": HEADERS,
        "body": json.dumps({"error": "Not found"}),
    }
