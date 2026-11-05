"""
Text-based emotion analysis using HuggingFace transformer classifiers.

Provides automatic emotion detection from text at the sentence or clause level,
mapping natural language to EmotionLabel instances with VAD vectors.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from tts.emotion.emotion_labels import (
    EmotionCategory,
    EmotionLabel,
    EmotionVector,
    get_default_vector,
)

logger = logging.getLogger(__name__)

_HF_LABEL_MAP = {
    "anger": EmotionCategory.ANGRY,
    "disgust": EmotionCategory.DISGUSTED,
    "fear": EmotionCategory.FEARFUL,
    "joy": EmotionCategory.HAPPY,
    "neutral": EmotionCategory.NEUTRAL,
    "sadness": EmotionCategory.SAD,
    "surprise": EmotionCategory.SURPRISED,
}


class TextEmotionAnalyzer:
    """
    Detect emotion from text using a pretrained transformer classifier.
    """

    DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.4,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        """Load the HuggingFace pipeline if not already loaded."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading emotion classifier: %s", self.model_name)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self._resolve_pipeline_device(self.device),
            )
            logger.info("Emotion classifier loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load emotion model: %s", exc)
            raise RuntimeError(
                f"Could not load emotion model '{self.model_name}'. "
                f"Make sure 'transformers' is installed and the model exists. "
                f"Error: {exc}"
            ) from exc

    def analyze(self, text: str) -> EmotionLabel:
        """
        Analyze a single piece of text and return an EmotionLabel.
        """
        self._ensure_loaded()

        if not text or not text.strip():
            return self._neutral_label()

        results = self._pipeline(text[:512])
        scores = results[0] if results else []
        if not scores:
            return self._neutral_label()

        scores.sort(key=lambda item: item["score"], reverse=True)
        top = scores[0]
        label_name = top["label"].lower().strip()
        confidence = float(top["score"])
        category = _HF_LABEL_MAP.get(label_name, EmotionCategory.NEUTRAL)

        if confidence < self.confidence_threshold:
            logger.debug(
                "Low confidence %.2f for '%s' -> falling back to NEUTRAL",
                confidence,
                label_name,
            )
            return self._neutral_label()

        return EmotionLabel(
            category=category,
            vector=get_default_vector(category, intensity=confidence),
            confidence=confidence,
        )

    def analyze_segments(self, segments: List[str]) -> List[EmotionLabel]:
        """Analyze multiple text segments in sequence."""
        return [self.analyze(segment) for segment in segments]

    @staticmethod
    def _neutral_label() -> EmotionLabel:
        return EmotionLabel(
            category=EmotionCategory.NEUTRAL,
            vector=EmotionVector.neutral(),
            confidence=1.0,
        )

    @staticmethod
    def _resolve_pipeline_device(device: str):
        """Normalize user-friendly device strings for transformers.pipeline."""
        normalized = device.strip().lower()
        if normalized == "cpu":
            return -1
        if normalized == "cuda":
            return 0
        if normalized.startswith("cuda:"):
            try:
                return int(normalized.split(":", 1)[1])
            except ValueError:
                logger.warning("Invalid CUDA device '%s'; defaulting to cuda:0", device)
                return 0
        return device
