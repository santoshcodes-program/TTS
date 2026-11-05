"""
No-key TTS provider built on edge-tts.

This uses the publicly available `edge-tts` Python library, which exposes the
same online text-to-speech service used by Microsoft Edge without requiring an
API key. Since custom expressive SSML styles are not supported by the service,
we approximate emotion through voice choice and prosody controls such as rate,
pitch, and volume.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tts.emotion.emotion_labels import EmotionCategory
from tts.emotion.simple_emotion_detector import EmotionDetectionResult


@dataclass(frozen=True)
class EdgeVoicePreset:
    """Voice and prosody controls used to approximate emotion."""

    voice: str
    rate: str
    pitch: str
    volume: str


@dataclass(frozen=True)
class EdgeSpeechSynthesisResult:
    """Synthesized audio and metadata for UI display."""

    audio_bytes: bytes
    mime_type: str
    voice_name: str
    rate: str
    pitch: str
    volume: str


_PRESETS = {
    EmotionCategory.NEUTRAL: EdgeVoicePreset(
        voice="en-US-AriaNeural", rate="+0%", pitch="+0Hz", volume="+0%"
    ),
    EmotionCategory.HAPPY: EdgeVoicePreset(
        voice="en-US-JennyNeural", rate="+14%", pitch="+24Hz", volume="+8%"
    ),
    EmotionCategory.SAD: EdgeVoicePreset(
        voice="en-US-AriaNeural", rate="-18%", pitch="-20Hz", volume="-6%"
    ),
    EmotionCategory.ANGRY: EdgeVoicePreset(
        voice="en-US-GuyNeural", rate="+18%", pitch="+18Hz", volume="+14%"
    ),
    EmotionCategory.FEARFUL: EdgeVoicePreset(
        voice="en-US-JennyNeural", rate="+10%", pitch="+30Hz", volume="-2%"
    ),
    EmotionCategory.SURPRISED: EdgeVoicePreset(
        voice="en-US-GuyNeural", rate="+20%", pitch="+34Hz", volume="+10%"
    ),
    EmotionCategory.DISGUSTED: EdgeVoicePreset(
        voice="en-US-AriaNeural", rate="-8%", pitch="-10Hz", volume="-4%"
    ),
}


def get_edge_preset(category: EmotionCategory, confidence: float = 1.0) -> EdgeVoicePreset:
    """
    Return a preset with slightly confidence-aware rate scaling.
    """
    base = _PRESETS[category]
    if category == EmotionCategory.NEUTRAL:
        return base

    scale = max(0.75, min(1.25, confidence + 0.35))
    return EdgeVoicePreset(
        voice=base.voice,
        rate=_scale_percent(base.rate, scale),
        pitch=base.pitch,
        volume=base.volume,
    )


class EdgeSpeechSynthesizer:
    """Synthesize MP3 speech using edge-tts without any API key."""

    def synthesize(self, text: str, detection: EmotionDetectionResult) -> EdgeSpeechSynthesisResult:
        preset = get_edge_preset(detection.category, detection.confidence)
        audio_bytes = _run_async(
            _synthesize_mp3(
                text=text,
                voice=preset.voice,
                rate=preset.rate,
                pitch=preset.pitch,
                volume=preset.volume,
            )
        )
        return EdgeSpeechSynthesisResult(
            audio_bytes=audio_bytes,
            mime_type="audio/mpeg",
            voice_name=preset.voice,
            rate=preset.rate,
            pitch=preset.pitch,
            volume=preset.volume,
        )


async def _synthesize_mp3(text: str, voice: str, rate: str, pitch: str, volume: str) -> bytes:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is not installed. Install it with: python -m pip install edge-tts"
        ) from exc

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )

    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])

    if not chunks:
        raise RuntimeError("edge-tts returned no audio.")

    return bytes(chunks)


def _run_async(coro):
    """
    Run an async coroutine from sync Streamlit code.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def _scale_percent(value: str, scale: float) -> str:
    sign = -1 if value.startswith("-") else 1
    stripped = value.replace("%", "").replace("+", "").replace("-", "")
    try:
        numeric = float(stripped)
    except ValueError:
        return value

    scaled = sign * numeric * scale
    prefix = "+" if scaled >= 0 else ""
    return f"{prefix}{scaled:.0f}%"
