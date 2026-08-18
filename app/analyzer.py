"""
Multi-model sentiment analysis engine.

Combines VADER, TextBlob, and optional Hugging Face Inference API
to classify text and score consistency across models.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Bundle NLTK data for serverless (Vercel/Netlify)
_nltk_dir = Path(__file__).resolve().parent.parent / "nltk_data"
if _nltk_dir.exists():
    os.environ.setdefault("NLTK_DATA", str(_nltk_dir))
    import nltk

    nltk.data.path.insert(0, str(_nltk_dir))

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEUTRAL_THRESHOLD = 0.05


@dataclass
class ModelResult:
    model: str
    label: str
    confidence: float
    scores: dict[str, float]


@dataclass
class AnalysisResult:
    text: str
    final_label: str
    final_confidence: float
    consistency_score: float
    reliability_score: float
    models: list[ModelResult]
    breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "final_label": self.final_label,
            "final_confidence": round(self.final_confidence, 4),
            "consistency_score": round(self.consistency_score, 2),
            "reliability_score": round(self.reliability_score, 2),
            "models": [asdict(m) for m in self.models],
            "breakdown": self.breakdown,
        }


def _compound_to_label(compound: float) -> str:
    if compound >= NEUTRAL_THRESHOLD:
        return "positive"
    if compound <= -NEUTRAL_THRESHOLD:
        return "negative"
    return "neutral"


def _polarity_to_label(polarity: float) -> str:
    if polarity > NEUTRAL_THRESHOLD:
        return "positive"
    if polarity < -NEUTRAL_THRESHOLD:
        return "negative"
    return "neutral"


def _normalize_scores(pos: float, neg: float, neu: float) -> dict[str, float]:
    total = pos + neg + neu
    if total == 0:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    return {
        "positive": round(pos / total, 4),
        "negative": round(neg / total, 4),
        "neutral": round(neu / total, 4),
    }


class SentimentAnalyzer:
    """Classifies text sentiment using multiple NLP approaches."""

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self._hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", "")
        self._hf_model = os.environ.get(
            "HUGGINGFACE_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    def analyze(self, text: str) -> AnalysisResult:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")

        model_results: list[ModelResult] = []
        model_results.append(self._analyze_vader(text))
        model_results.append(self._analyze_textblob(text))

        hf_result = self._analyze_huggingface(text)
        if hf_result:
            model_results.append(hf_result)

        final_label, final_confidence = self._ensemble_vote(model_results)
        consistency = self._compute_consistency(model_results)
        reliability = self._compute_reliability(model_results, consistency, text)

        breakdown = {
            "agreement": self._label_agreement(model_results),
            "dominant_sentiment": final_label,
            "text_length": len(text),
            "word_count": len(re.findall(r"\b\w+\b", text)),
            "models_used": len(model_results),
        }

        return AnalysisResult(
            text=text,
            final_label=final_label,
            final_confidence=final_confidence,
            consistency_score=consistency,
            reliability_score=reliability,
            models=model_results,
            breakdown=breakdown,
        )

    def analyze_batch(self, texts: list[str]) -> list[AnalysisResult]:
        return [self.analyze(t) for t in texts if t.strip()]

    def _analyze_vader(self, text: str) -> ModelResult:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        label = _compound_to_label(compound)
        confidence = min(abs(compound), 1.0)

        return ModelResult(
            model="VADER",
            label=label,
            confidence=round(confidence, 4),
            scores={
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "compound": round(compound, 4),
            },
        )

    def _analyze_textblob(self, text: str) -> ModelResult:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label = _polarity_to_label(polarity)
        confidence = min(abs(polarity) + subjectivity * 0.3, 1.0)

        pos = max(polarity, 0)
        neg = max(-polarity, 0)
        neu = 1.0 - pos - neg if polarity != 0 else 1.0

        return ModelResult(
            model="TextBlob",
            label=label,
            confidence=round(confidence, 4),
            scores=_normalize_scores(pos, neg, neu)
            | {"polarity": round(polarity, 4), "subjectivity": round(subjectivity, 4)},
        )

    def _analyze_huggingface(self, text: str) -> ModelResult | None:
        if not self._hf_token:
            return None

        try:
            import urllib.error
            import urllib.request
            import json

            url = f"https://api-inference.huggingface.co/models/{self._hf_model}"
            payload = json.dumps({"inputs": text[:512]}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._hf_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if isinstance(data, list) and data and isinstance(data[0], list):
                items = sorted(data[0], key=lambda x: x["score"], reverse=True)
                top = items[0]
                raw_label = top["label"].lower()
                label_map = {
                    "positive": "positive",
                    "negative": "negative",
                    "neutral": "neutral",
                    "label_0": "negative",
                    "label_1": "neutral",
                    "label_2": "positive",
                }
                label = label_map.get(raw_label, raw_label)
                scores = {item["label"].lower(): round(item["score"], 4) for item in items}

                return ModelResult(
                    model="HuggingFace",
                    label=label,
                    confidence=round(top["score"], 4),
                    scores=scores,
                )
        except Exception:
            return None

        return None

    def _ensemble_vote(self, results: list[ModelResult]) -> tuple[str, float]:
        votes: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for result in results:
            votes[result.label] += result.confidence

        final_label = max(votes, key=votes.get)  # type: ignore[arg-type]
        total = sum(votes.values()) or 1.0
        final_confidence = votes[final_label] / total

        return final_label, final_confidence

    def _compute_consistency(self, results: list[ModelResult]) -> float:
        """Score 0-100 based on how much models agree on the label."""
        if len(results) < 2:
            return 100.0

        labels = [r.label for r in results]
        most_common = max(set(labels), key=labels.count)
        agreement_count = labels.count(most_common)
        base_score = (agreement_count / len(labels)) * 100

        confidences = [r.confidence for r in results if r.label == most_common]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

        return min(base_score * (0.7 + 0.3 * avg_conf), 100.0)

    def _compute_reliability(
        self, results: list[ModelResult], consistency: float, text: str
    ) -> float:
        """
        Proxy for output accuracy: combines model agreement, confidence,
        and text quality signals (length, not too short).
        """
        avg_confidence = sum(r.confidence for r in results) / len(results)
        word_count = len(re.findall(r"\b\w+\b", text))

        length_factor = 1.0
        if word_count < 3:
            length_factor = 0.5
        elif word_count < 8:
            length_factor = 0.75

        reliability = (
            consistency * 0.45 + avg_confidence * 100 * 0.35 + length_factor * 100 * 0.20
        )
        return min(reliability, 100.0)

    def _label_agreement(self, results: list[ModelResult]) -> dict[str, int]:
        counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r.label] = counts.get(r.label, 0) + 1
        return counts
