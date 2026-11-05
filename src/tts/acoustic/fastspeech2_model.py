"""
Emotion-aware FastSpeech2 acoustic model wrapper.

Wraps the ESPnet2 FastSpeech2 pretrained model and adds post-hoc
emotion conditioning to the generated mel spectrograms.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from tts.acoustic.emotion_conditioning import EmotionConditioner
from tts.emotion.emotion_labels import EmotionVector

logger = logging.getLogger(__name__)


class EmotionAwareFastSpeech2:
    """
    TTS acoustic model that generates mel spectrograms from text
    with optional emotion conditioning.
    """

    DEFAULT_MODEL_TAG = "kan-bayashi/ljspeech_fastspeech2"

    def __init__(
        self,
        model_tag: Optional[str] = None,
        device: str = "cpu",
        pitch_range: float = 3.0,
        energy_range: float = 0.5,
        rate_range: float = 0.3,
        smoothing_window: int = 5,
    ) -> None:
        self.model_tag = model_tag or self.DEFAULT_MODEL_TAG
        self.device = device
        self._tts = None
        self._conditioner = EmotionConditioner(
            pitch_range=pitch_range,
            energy_range=energy_range,
            rate_range=rate_range,
            smoothing_window=smoothing_window,
        )

    def _ensure_loaded(self) -> None:
        """Load the ESPnet2 TTS model if not already loaded."""
        if self._tts is not None:
            return

        try:
            from espnet2.bin.tts_inference import Text2Speech

            logger.info("Loading FastSpeech2 model: %s", self.model_tag)
            self._tts = Text2Speech.from_pretrained(
                model_tag=self.model_tag,
                device=self.device,
            )
            logger.info("FastSpeech2 model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load FastSpeech2: %s", exc)
            raise RuntimeError(
                f"Could not load FastSpeech2 model '{self.model_tag}'. "
                f"Ensure 'espnet' and 'espnet_model_zoo' are installed. "
                f"Error: {exc}"
            ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    def synthesize(
        self,
        text: str,
        emotion: Optional[EmotionVector] = None,
    ) -> np.ndarray:
        """
        Synthesize a mel spectrogram from text.
        """
        self._ensure_loaded()

        with torch.no_grad():
            output = self._tts(text)

        mel = output["feat_gen"].cpu().numpy()
        if emotion is not None:
            mel = self._conditioner.condition_mel_static(mel, emotion)
        return mel

    def synthesize_with_curve(
        self,
        text: str,
        emotion_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize mel spectrogram with per-frame emotion conditioning.
        """
        self._ensure_loaded()

        with torch.no_grad():
            output = self._tts(text)

        mel = output["feat_gen"].cpu().numpy()
        return self._conditioner.condition_mel_dynamic(mel, emotion_frames)

    def save_mel(self, mel: np.ndarray, path: str) -> None:
        """Save mel spectrogram to a `.npy` file."""
        np.save(path, mel)
        logger.info("Mel spectrogram saved to %s (shape: %s)", path, mel.shape)
