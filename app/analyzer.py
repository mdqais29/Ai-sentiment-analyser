"""
Multi-model sentiment analysis engine.

Combines VADER, lexicon-based NLP, and optional Hugging Face Inference API
to classify text and score consistency across models.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEUTRAL_THRESHOLD = 0.05

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "loved", "loving",
    "best", "happy", "fantastic", "wonderful", "brilliant", "perfect", "nice",
    "thrilled", "delighted", "outstanding", "superb", "beautiful", "enjoy",
    "enjoyed", "recommend", "recommended", "impressive", "satisfied", "pleased",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "hated", "poor",
    "disappointing", "disappointed", "angry", "upset", "useless", "broken",
    "waste", "wasted", "fail", "failed", "ugly", "slow", "rude", "unhappy",
    "frustrating", "frustrated", "annoying", "annoyed", "disgusting",
}


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


def _score_to_label(score: float) -> str:
    if score >= NEUTRAL_THRESHOLD:
        return "positive"
    if score <= -NEUTRAL_THRESHOLD:
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

        model_results: list[ModelResult] = [
            self._analyze_vader(text),
            self._analyze_lexicon(text),
        ]

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
        label = _score_to_label(compound)
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

    def _analyze_lexicon(self, text: str) -> ModelResult:
        words = {w.strip(".,!?;:'\"").lower() for w in re.findall(r"\b\w+\b", text)}
        pos_hits = len(words & POSITIVE_WORDS)
        neg_hits = len(words & NEGATIVE_WORDS)
        total_hits = pos_hits + neg_hits

        if total_hits == 0:
            score = 0.0
            confidence = 0.35
        else:
            score = (pos_hits - neg_hits) / total_hits
            confidence = min(abs(score) + total_hits * 0.1, 1.0)

        label = _score_to_label(score)
        pos = max(score, 0)
        neg = max(-score, 0)
        neu = max(1.0 - pos - neg, 0.0)

        return ModelResult(
            model="Lexicon",
            label=label,
            confidence=round(confidence, 4),
            scores=_normalize_scores(pos, neg, neu)
            | {
                "positive_hits": pos_hits,
                "negative_hits": neg_hits,
                "score": round(score, 4),
            },
        )

    def _analyze_huggingface(self, text: str) -> ModelResult | None:
        if not self._hf_token:
            return None

        try:
            import json
            import urllib.request

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
