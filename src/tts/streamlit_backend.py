"""
Helpers for the Streamlit demo.

These utilities keep UI concerns out of the core pipeline while still
reusing the same text -> mel -> audio stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Union
import wave

import numpy as np

from tts.emotion.emotion_labels import EmotionCategory, EmotionVector
from tts.pipeline import EmotionTTSPipeline

EmotionInput = Optional[Union[str, EmotionCategory, EmotionVector]]


@dataclass
class SynthesisArtifacts:
    """Intermediate and final outputs produced for the Streamlit app."""

    text: str
    mel: np.ndarray
    waveform: np.ndarray
    sample_rate: int
    emotion_name: str = "neutral"
    emotion_source: str = "manual"
    emotion_confidence: Optional[float] = None


@dataclass(frozen=True)
class ResolvedEmotion:
    """Resolved emotion request for a single preview synthesis."""

    emotion_input: EmotionInput
    intensity: float
    display_name: str
    source: str
    confidence: Optional[float] = None


def synthesize_preview(
    pipeline: EmotionTTSPipeline,
    text: str,
    emotion: EmotionInput = None,
    intensity: float = 1.0,
) -> SynthesisArtifacts:
    """
    Run the cleaned staged pipeline and return both mel and audio outputs.
    """
    mel = pipeline.text_to_mel(text, emotion=emotion, intensity=intensity)
    waveform = pipeline.mel_to_audio(mel)
    return SynthesisArtifacts(
        text=text,
        mel=mel,
        waveform=waveform,
        sample_rate=pipeline.config.sample_rate,
    )


def resolve_emotion_for_preview(
    pipeline: EmotionTTSPipeline,
    text: str,
    emotion_mode: str = "auto",
    manual_emotion: Optional[Union[str, EmotionCategory]] = None,
    intensity: float = 1.0,
) -> ResolvedEmotion:
    """
    Resolve a UI emotion request into a concrete pipeline input.

    Modes:
    - ``auto``: run the text classifier once and pass the detected vector
    - ``manual``: use the user-selected emotion label
    - ``neutral`` / ``off`` / ``none``: explicitly disable emotional coloring
    """
    mode = emotion_mode.strip().lower()

    if mode == "manual":
        if manual_emotion is None:
            raise ValueError("A manual emotion must be provided when emotion_mode='manual'.")

        category = (
            manual_emotion
            if isinstance(manual_emotion, EmotionCategory)
            else EmotionCategory.from_string(str(manual_emotion))
        )
        return ResolvedEmotion(
            emotion_input=category.value,
            intensity=float(intensity),
            display_name=category.value,
            source="manual",
        )

    if mode == "auto":
        label = pipeline.emotion_analyzer.analyze(text)
        if label.is_reliable(pipeline.config.emotion_confidence_threshold):
            return ResolvedEmotion(
                emotion_input=label.vector.scaled(float(intensity)),
                intensity=1.0,
                display_name=label.category.value,
                source="auto",
                confidence=label.confidence,
            )

        return ResolvedEmotion(
            emotion_input=EmotionCategory.NEUTRAL.value,
            intensity=1.0,
            display_name=EmotionCategory.NEUTRAL.value,
            source="auto-low-confidence",
            confidence=label.confidence,
        )

    if mode in {"neutral", "off", "none", "disabled"}:
        return ResolvedEmotion(
            emotion_input=EmotionCategory.NEUTRAL.value,
            intensity=1.0,
            display_name=EmotionCategory.NEUTRAL.value,
            source="neutral",
        )

    raise ValueError(f"Unknown emotion mode: {emotion_mode}")


def synthesize_emotional_preview(
    pipeline: EmotionTTSPipeline,
    text: str,
    emotion_mode: str = "auto",
    manual_emotion: Optional[Union[str, EmotionCategory]] = None,
    intensity: float = 1.0,
) -> SynthesisArtifacts:
    """
    Synthesize preview audio while also reporting how emotion was resolved.
    """
    resolved = resolve_emotion_for_preview(
        pipeline,
        text=text,
        emotion_mode=emotion_mode,
        manual_emotion=manual_emotion,
        intensity=intensity,
    )

    artifacts = synthesize_preview(
        pipeline,
        text=text,
        emotion=resolved.emotion_input,
        intensity=resolved.intensity,
    )
    artifacts.emotion_name = resolved.display_name
    artifacts.emotion_source = resolved.source
    artifacts.emotion_confidence = resolved.confidence
    return artifacts


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """
    Encode a waveform as in-memory WAV bytes for Streamlit playback/download.
    """
    buffer = BytesIO()
    write_wave_file(buffer, waveform, sample_rate)
    return buffer.getvalue()


def mel_to_display_image(mel: np.ndarray) -> np.ndarray:
    """
    Convert a mel spectrogram into a simple RGB heatmap image.
    """
    mel_array = np.asarray(mel, dtype=np.float32)
    if mel_array.ndim != 2:
        raise ValueError(f"Expected a 2D mel spectrogram, got shape {mel_array.shape}.")

    if mel_array.shape[0] == 80 and mel_array.shape[1] != 80:
        mel_array = mel_array.T

    mel_view = np.flipud(mel_array.T)
    mel_view = mel_view - float(mel_view.min())
    peak = float(mel_view.max())
    if peak > 0.0:
        mel_view = mel_view / peak

    mel_view = np.power(mel_view, 0.8)

    red = np.clip(255.0 * (0.12 + 1.15 * mel_view), 0.0, 255.0)
    green = np.clip(255.0 * (0.08 + 0.95 * np.sqrt(mel_view)), 0.0, 255.0)
    blue = np.clip(255.0 * (0.24 + 0.60 * (1.0 - mel_view)), 0.0, 255.0)

    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def write_wave_file(target, waveform: np.ndarray, sample_rate: int) -> None:
    """
    Write a mono PCM16 WAV file to a path or file-like object.
    """
    pcm16 = waveform_to_pcm16(np.asarray(waveform, dtype=np.float32))
    with wave.open(target, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def waveform_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a float waveform in [-1, 1] to signed 16-bit PCM.
    """
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)
