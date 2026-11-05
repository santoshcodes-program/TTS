"""
Emotion labels, vectors, and mappings for the emotion-aware TTS system.

Emotions are modeled as both discrete categories and continuous
Valence-Arousal-Dominance (VAD) vectors, enabling fine-grained
intensity control and smooth interpolation between emotional states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class EmotionCategory(Enum):
    """Discrete emotion categories supported by the TTS system."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"

    @classmethod
    def from_string(cls, name: str) -> "EmotionCategory":
        """Case-insensitive lookup by value or name."""
        name_lower = name.strip().lower()
        for member in cls:
            if member.value == name_lower or member.name.lower() == name_lower:
                return member
        raise ValueError(
            f"Unknown emotion: '{name}'. "
            f"Valid: {[e.value for e in cls]}"
        )


@dataclass
class EmotionVector:
    """
    Continuous emotion representation in the VAD (Valence-Arousal-Dominance)
    space plus an intensity multiplier.

    - valence:   pleasure/displeasure    (0.0 = very negative, 1.0 = very positive)
    - arousal:   calm/excited             (0.0 = very calm,     1.0 = very excited)
    - dominance: submissive/dominant      (0.0 = submissive,    1.0 = dominant)
    - intensity: overall emotion strength (0.0 = neutral,       1.0 = maximum)
    """

    valence: float = 0.5
    arousal: float = 0.5
    dominance: float = 0.5
    intensity: float = 0.5

    def __post_init__(self) -> None:
        """Clamp all values to [0, 1]."""
        self.valence = max(0.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.intensity = max(0.0, min(1.0, self.intensity))

    def to_list(self) -> list[float]:
        """Return [valence, arousal, dominance, intensity] as a list."""
        return [self.valence, self.arousal, self.dominance, self.intensity]

    def scaled(self, factor: float) -> "EmotionVector":
        """Return a new vector with intensity scaled by *factor*."""
        return EmotionVector(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            intensity=self.intensity * factor,
        )

    @classmethod
    def neutral(cls) -> "EmotionVector":
        """Default neutral emotion vector."""
        return cls(valence=0.5, arousal=0.3, dominance=0.5, intensity=0.0)

    @classmethod
    def lerp(cls, a: "EmotionVector", b: "EmotionVector", t: float) -> "EmotionVector":
        """Linear interpolation between two vectors at ratio *t* ∈ [0, 1]."""
        t = max(0.0, min(1.0, t))
        return cls(
            valence=a.valence + (b.valence - a.valence) * t,
            arousal=a.arousal + (b.arousal - a.arousal) * t,
            dominance=a.dominance + (b.dominance - a.dominance) * t,
            intensity=a.intensity + (b.intensity - a.intensity) * t,
        )


@dataclass
class EmotionLabel:
    """
    A complete emotion annotation: discrete category + continuous vector
    + detection confidence.
    """

    category: EmotionCategory
    vector: EmotionVector
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))

    def is_reliable(self, threshold: float = 0.4) -> bool:
        """Return True if detection confidence exceeds *threshold*."""
        return self.confidence >= threshold


# ─── Default VAD mappings ────────────────────────────────────────────────────

# Based on psychological research (Russell's Circumplex Model + PAD model).
# Format: EmotionCategory → EmotionVector(valence, arousal, dominance, intensity)

DEFAULT_VAD_MAP: Dict[EmotionCategory, EmotionVector] = {
    EmotionCategory.NEUTRAL:   EmotionVector(0.50, 0.30, 0.50, 0.0),
    EmotionCategory.HAPPY:     EmotionVector(0.85, 0.70, 0.65, 0.8),
    EmotionCategory.SAD:       EmotionVector(0.15, 0.25, 0.25, 0.7),
    EmotionCategory.ANGRY:     EmotionVector(0.15, 0.85, 0.80, 0.9),
    EmotionCategory.FEARFUL:   EmotionVector(0.20, 0.80, 0.15, 0.8),
    EmotionCategory.SURPRISED: EmotionVector(0.60, 0.85, 0.40, 0.7),
    EmotionCategory.DISGUSTED: EmotionVector(0.15, 0.55, 0.60, 0.7),
}


def get_default_vector(category: EmotionCategory, intensity: float = 1.0) -> EmotionVector:
    """
    Look up the default VAD vector for a category,
    optionally scaled by *intensity*.
    """
    base = DEFAULT_VAD_MAP.get(category, EmotionVector.neutral())
    return base.scaled(intensity) if intensity != 1.0 else EmotionVector(
        valence=base.valence,
        arousal=base.arousal,
        dominance=base.dominance,
        intensity=base.intensity,
    )
