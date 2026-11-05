"""
Lightweight text emotion detection for the Streamlit demo.

This module provides a dependency-light emotion detector that can run with
simple keyword heuristics, while optionally upgrading to the existing
transformer-based analyzer when the dependency is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional

from tts.emotion.emotion_labels import EmotionCategory


_TOKEN_RE = re.compile(r"[a-zA-Z']+")

_KEYWORDS: Dict[EmotionCategory, set[str]] = {
    EmotionCategory.HAPPY: {
        "amazing", "awesome", "celebrate", "cheerful", "delighted", "excited",
        "fantastic", "glad", "great", "happy", "joy", "joyful", "love",
        "smile", "wonderful",
    },
    EmotionCategory.SAD: {
        "alone", "broken", "cry", "crying", "depressed", "grief", "heartbroken",
        "lonely", "loss", "lost", "miss", "miserable", "sad", "sorry", "tears",
        "upset",
    },
    EmotionCategory.ANGRY: {
        "annoyed", "angry", "furious", "hate", "mad", "rage", "ridiculous",
        "stupid", "terrible", "unacceptable", "useless", "worst",
    },
    EmotionCategory.FEARFUL: {
        "afraid", "anxious", "danger", "fear", "nervous", "panic", "scared",
        "terrified", "threat", "unsafe", "worried",
    },
    EmotionCategory.SURPRISED: {
        "amazed", "astonishing", "incredible", "really", "shocked", "suddenly",
        "surprised", "unbelievable", "unexpected", "wow",
    },
    EmotionCategory.DISGUSTED: {
        "awful", "disgusting", "gross", "nasty", "revolting", "sickening",
        "vile", "yuck",
    },
}

_INTENSIFIERS = {
    "absolutely", "extremely", "incredibly", "really", "so", "super", "too", "very",
}

_NEGATORS = {"barely", "hardly", "never", "no", "not"}


@dataclass(frozen=True)
class EmotionDetectionResult:
    """Structured emotion output for UI display and TTS mapping."""

    category: EmotionCategory
    confidence: float
    scores: Dict[str, float]
    source: str
    evidence: str = ""


class EmotionDetector:
    """
    Detect coarse text emotion labels with optional transformer enhancement.

    If the Hugging Face stack is available, it uses the project's existing
    `TextEmotionAnalyzer`. Otherwise, it falls back to keyword heuristics.
    """

    def __init__(self, prefer_transformers: bool = True, confidence_threshold: float = 0.4) -> None:
        self.prefer_transformers = prefer_transformers
        self.confidence_threshold = confidence_threshold
        self._hf_analyzer = None
        self._hf_failed = False

    def detect(self, text: str) -> EmotionDetectionResult:
        stripped = text.strip()
        if not stripped:
            return self._neutral_result(source="empty")

        if self.prefer_transformers:
            transformer_result = self._detect_with_transformers(stripped)
            if transformer_result is not None:
                return transformer_result

        return self._detect_with_keywords(stripped)

    def _detect_with_transformers(self, text: str) -> Optional[EmotionDetectionResult]:
        if self._hf_failed:
            return None

        try:
            if self._hf_analyzer is None:
                from tts.emotion.text_emotion_analyzer import TextEmotionAnalyzer

                self._hf_analyzer = TextEmotionAnalyzer(
                    confidence_threshold=self.confidence_threshold,
                )

            label = self._hf_analyzer.analyze(text)
            return EmotionDetectionResult(
                category=label.category,
                confidence=label.confidence,
                scores={label.category.value: label.confidence},
                source="transformers",
                evidence="transformer classifier",
            )
        except Exception:
            self._hf_failed = True
            return None

    def _detect_with_keywords(self, text: str) -> EmotionDetectionResult:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        if not tokens:
            return self._neutral_result(source="heuristic")

        scores = {category: 0.05 for category in EmotionCategory}
        scores[EmotionCategory.NEUTRAL] = 0.2
        matched_words: list[str] = []

        token_count = len(tokens)
        for index, token in enumerate(tokens):
            is_intensified = index > 0 and tokens[index - 1] in _INTENSIFIERS
            is_negated = index > 0 and tokens[index - 1] in _NEGATORS
            weight = 1.0 + (0.35 if is_intensified else 0.0)

            for category, keywords in _KEYWORDS.items():
                if token not in keywords:
                    continue

                matched_words.append(token)
                if is_negated:
                    scores[EmotionCategory.NEUTRAL] += 0.2
                    scores[category] += 0.15
                else:
                    scores[category] += weight

        exclamations = text.count("!")
        questions = text.count("?")
        uppercase_ratio = sum(1 for char in text if char.isupper()) / max(len(text), 1)

        if exclamations:
            scores[EmotionCategory.SURPRISED] += 0.2 * exclamations
            scores[EmotionCategory.HAPPY] += 0.1 * exclamations
            scores[EmotionCategory.ANGRY] += 0.1 * exclamations
        if questions:
            scores[EmotionCategory.SURPRISED] += 0.12 * questions
            scores[EmotionCategory.FEARFUL] += 0.05 * questions
        if uppercase_ratio > 0.18 and token_count > 3:
            scores[EmotionCategory.ANGRY] += 0.35
            scores[EmotionCategory.SURPRISED] += 0.15

        if any(word in tokens for word in {"love", "great", "wonderful", "amazing"}):
            scores[EmotionCategory.HAPPY] += 0.3
        if any(word in tokens for word in {"hate", "worst", "ridiculous"}):
            scores[EmotionCategory.ANGRY] += 0.35
        if any(word in tokens for word in {"miss", "lost", "lonely"}):
            scores[EmotionCategory.SAD] += 0.25
        if any(word in tokens for word in {"afraid", "worried", "panic"}):
            scores[EmotionCategory.FEARFUL] += 0.3

        strongest = max(scores, key=scores.get)
        total = sum(scores.values())
        normalized_scores = {
            category.value: round(scores[category] / total, 4)
            for category in EmotionCategory
        }
        confidence = normalized_scores[strongest.value]

        if strongest != EmotionCategory.NEUTRAL and confidence < 0.24:
            strongest = EmotionCategory.NEUTRAL
            confidence = normalized_scores[EmotionCategory.NEUTRAL.value]

        evidence = ", ".join(sorted(set(matched_words))[:5]) if matched_words else "tone and punctuation"
        return EmotionDetectionResult(
            category=strongest,
            confidence=confidence,
            scores=normalized_scores,
            source="heuristic",
            evidence=evidence,
        )

    @staticmethod
    def _neutral_result(source: str) -> EmotionDetectionResult:
        return EmotionDetectionResult(
            category=EmotionCategory.NEUTRAL,
            confidence=1.0,
            scores={EmotionCategory.NEUTRAL.value: 1.0},
            source=source,
            evidence="empty text",
        )
