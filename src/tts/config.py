"""
Central configuration for the emotion-aware TTS system.

All paths, model tags, device settings, and emotion defaults
are managed here to keep the pipeline clean and configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tts.emotion.emotion_labels import EmotionCategory

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """
    Configuration for the EmotionTTSPipeline.

    All fields have sensible defaults that work out of the box
    (assuming the standard project layout and model files).
    """

    # ── Device ────────────────────────────────────────────────────
    device: str = "cpu"

    # ── FastSpeech2 acoustic model ───────────────────────────────
    fastspeech2_model_tag: str = "kan-bayashi/ljspeech_fastspeech2"

    # ── HiFi-GAN vocoder ─────────────────────────────────────────
    vocoder_checkpoint_path: str = "hifigan.v1/checkpoint-400000steps.pkl"

    # ── Emotion analysis ─────────────────────────────────────────
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_confidence_threshold: float = 0.4
    enable_emotion_conditioning: bool = True

    # ── Default emotion ──────────────────────────────────────────
    default_emotion: str = "neutral"
    default_intensity: float = 0.7

    # ── Audio output ─────────────────────────────────────────────
    sample_rate: int = 22050

    # ── Emotion conditioning parameters ──────────────────────────
    pitch_range: float = 3.0      # max semitones shift
    energy_range: float = 0.5     # max energy offset
    rate_range: float = 0.3       # max rate offset
    smoothing_window: int = 5     # frames for conditioning smoothing

    # ── Text processing ──────────────────────────────────────────
    segmentation_level: str = "sentence"  # "sentence" or "clause"

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        self.default_emotion = self.default_emotion.lower().strip()
        self.segmentation_level = self.segmentation_level.lower().strip()
        assert self.segmentation_level in ("sentence", "clause"), \
            f"segmentation_level must be 'sentence' or 'clause', got '{self.segmentation_level}'"

    @property
    def default_emotion_category(self) -> EmotionCategory:
        return EmotionCategory.from_string(self.default_emotion)

    @classmethod
    def from_yaml(cls, path: str) -> "TTSConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load config files. Install it with 'python -m pip install pyyaml'."
            ) from exc

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        logger.info("Loaded config from %s", path)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to save config files. Install it with 'python -m pip install pyyaml'."
            ) from exc

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

        logger.info("Saved config to %s", path)

    def setup_logging(self) -> None:
        """Configure logging based on the config."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
