"""
SSML-like parser for emotion-annotated text markup.

Supports custom <emotion> tags that let users explicitly control
which emotion is applied to which text span, with optional intensity.

Example markup::

    <speak>
      <emotion category="happy" intensity="0.9">
        What a wonderful day!
      </emotion>
      <emotion category="sad" intensity="0.6">
        But I miss you so much.
      </emotion>
    </speak>
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

from tts.emotion.emotion_labels import (
    EmotionCategory,
    EmotionVector,
    get_default_vector,
)


@dataclass
class AnnotatedSegment:
    """
    A text segment with an optional emotion annotation.

    If *emotion_override* is None, the pipeline should use automatic
    emotion detection for this segment.
    """

    text: str
    emotion_category: Optional[EmotionCategory] = None
    emotion_intensity: float = 1.0
    emotion_override: Optional[EmotionVector] = None

    @property
    def has_emotion(self) -> bool:
        return self.emotion_override is not None or self.emotion_category is not None

    def get_vector(self) -> Optional[EmotionVector]:
        """Return the emotion vector, computing from category if needed."""
        if self.emotion_override is not None:
            return self.emotion_override
        if self.emotion_category is not None:
            return get_default_vector(self.emotion_category, self.emotion_intensity)
        return None


class SSMLParser:
    """
    Parses emotion-annotated SSML-like markup into a list of
    AnnotatedSegment objects.

    Supports:
    - ``<speak>`` root wrapper (optional)
    - ``<emotion category="..." intensity="...">text</emotion>`` tags
    - ``<break time="..."/>`` pause markers (stored as empty segments)
    - Plain text without any tags (treated as auto-detect)
    """

    # Quick check: does the text contain any XML-like tags?
    _TAG_RE = re.compile(r'<\s*(?:speak|emotion|break)\b', re.IGNORECASE)

    def parse(self, text: str) -> List[AnnotatedSegment]:
        """
        Parse input text (plain or SSML-annotated) into segments.

        If the text contains no recognized tags, it is returned as a
        single segment with no emotion override.
        """
        if not text or not text.strip():
            return []

        # If no tags detected, return as plain text
        if not self._TAG_RE.search(text):
            return [AnnotatedSegment(text=text.strip())]

        return self._parse_xml(text)

    def _parse_xml(self, text: str) -> List[AnnotatedSegment]:
        """Parse XML-structured SSML text."""
        # Wrap in <speak> if not already wrapped
        stripped = text.strip()
        if not stripped.lower().startswith("<speak"):
            stripped = f"<speak>{stripped}</speak>"

        try:
            root = ET.fromstring(stripped)
        except ET.ParseError as e:
            # Fallback: treat as plain text if XML is malformed
            return [AnnotatedSegment(text=text.strip())]

        segments: List[AnnotatedSegment] = []

        # Process root text (before first child)
        if root.text and root.text.strip():
            segments.append(AnnotatedSegment(text=root.text.strip()))

        for child in root:
            tag = child.tag.lower()

            if tag == "emotion":
                segment = self._parse_emotion_tag(child)
                if segment:
                    segments.append(segment)

            elif tag == "break":
                # Break/pause marker — stored as empty segment
                time_str = child.get("time", "0.5s")
                segments.append(AnnotatedSegment(
                    text="",
                    emotion_category=EmotionCategory.NEUTRAL,
                ))

            # Process tail text (text after this child, before next)
            if child.tail and child.tail.strip():
                segments.append(AnnotatedSegment(text=child.tail.strip()))

        return segments if segments else [AnnotatedSegment(text=text.strip())]

    def _parse_emotion_tag(self, element: ET.Element) -> Optional[AnnotatedSegment]:
        """Extract emotion annotation from an <emotion> element."""
        inner_text = (element.text or "").strip()
        if not inner_text:
            return None

        # Parse attributes
        category_str = element.get("category", "neutral").strip().lower()
        intensity_str = element.get("intensity", "1.0").strip()

        try:
            category = EmotionCategory.from_string(category_str)
        except ValueError:
            category = EmotionCategory.NEUTRAL

        try:
            intensity = float(intensity_str)
            intensity = max(0.0, min(1.0, intensity))
        except ValueError:
            intensity = 1.0

        # Optional VAD overrides
        valence = self._parse_float_attr(element, "valence")
        arousal = self._parse_float_attr(element, "arousal")
        dominance = self._parse_float_attr(element, "dominance")

        if valence is not None or arousal is not None or dominance is not None:
            # Explicit VAD override
            base = get_default_vector(category)
            override = EmotionVector(
                valence=valence if valence is not None else base.valence,
                arousal=arousal if arousal is not None else base.arousal,
                dominance=dominance if dominance is not None else base.dominance,
                intensity=intensity,
            )
        else:
            override = get_default_vector(category, intensity)

        return AnnotatedSegment(
            text=inner_text,
            emotion_category=category,
            emotion_intensity=intensity,
            emotion_override=override,
        )

    @staticmethod
    def _parse_float_attr(element: ET.Element, name: str) -> Optional[float]:
        """Parse an optional float attribute, returning None if absent."""
        val = element.get(name)
        if val is None:
            return None
        try:
            return max(0.0, min(1.0, float(val.strip())))
        except ValueError:
            return None
