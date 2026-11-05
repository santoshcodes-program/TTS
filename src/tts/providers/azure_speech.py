"""
Azure Speech provider for emotion-aware TTS.

This module uses Azure Speech SSML speaking styles so the output voice can
sound cheerful, sad, angry, excited, and so on based on detected text emotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Optional

from tts.emotion.emotion_labels import EmotionCategory
from tts.emotion.simple_emotion_detector import EmotionDetectionResult


@dataclass(frozen=True)
class AzureSpeechCredentials:
    """Credentials required for Azure Speech synthesis."""

    key: str
    region: str


@dataclass(frozen=True)
class VoiceStylePreset:
    """SSML style controls used to shape the synthesized voice."""

    style: str
    style_degree: float
    rate: str = "0%"
    pitch: str = "0%"
    volume: str = "0%"


@dataclass(frozen=True)
class AzureSpeechSynthesisResult:
    """Audio plus synthesis metadata returned by the Azure provider."""

    wav_bytes: bytes
    voice_name: str
    style: str
    ssml: str


_STYLE_PRESETS = {
    EmotionCategory.NEUTRAL: VoiceStylePreset("general", 1.0, rate="0%", pitch="0%"),
    EmotionCategory.HAPPY: VoiceStylePreset("cheerful", 1.45, rate="+6.0%", pitch="+4.0%"),
    EmotionCategory.SAD: VoiceStylePreset("sad", 1.25, rate="-10.0%", pitch="-3.0%"),
    EmotionCategory.ANGRY: VoiceStylePreset("angry", 1.35, rate="+7.0%", pitch="+2.0%"),
    EmotionCategory.FEARFUL: VoiceStylePreset("terrified", 1.4, rate="+9.0%", pitch="+5.0%"),
    EmotionCategory.SURPRISED: VoiceStylePreset("excited", 1.35, rate="+8.0%", pitch="+6.0%"),
    EmotionCategory.DISGUSTED: VoiceStylePreset("unfriendly", 1.2, rate="-2.0%", pitch="-4.0%"),
}


def get_style_preset(category: EmotionCategory, confidence: float = 1.0) -> VoiceStylePreset:
    """
    Return a voice style preset, scaled slightly by confidence.
    """
    base = _STYLE_PRESETS[category]
    scaled_degree = min(2.0, max(0.6, base.style_degree * max(0.75, confidence + 0.25)))
    return VoiceStylePreset(
        style=base.style,
        style_degree=scaled_degree,
        rate=base.rate,
        pitch=base.pitch,
        volume=base.volume,
    )


def build_emotional_ssml(
    text: str,
    detection: EmotionDetectionResult,
    voice_name: str = "en-US-JennyNeural",
) -> str:
    """
    Build SSML that maps the detected emotion to Azure speaking styles.
    """
    preset = get_style_preset(detection.category, detection.confidence)
    escaped_text = html.escape(text, quote=False)
    escaped_voice = html.escape(voice_name, quote=True)

    if preset.style == "general":
        express_open = ""
        express_close = ""
    else:
        express_open = (
            f'<mstts:express-as style="{preset.style}" '
            f'styledegree="{preset.style_degree:.2f}">'
        )
        express_close = "</mstts:express-as>"

    return (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" '
        'xml:lang="en-US">'
        f'<voice name="{escaped_voice}">'
        f"{express_open}"
        f'<prosody rate="{preset.rate}" pitch="{preset.pitch}" volume="{preset.volume}">'
        f"{escaped_text}"
        "</prosody>"
        f"{express_close}"
        "</voice>"
        "</speak>"
    )


class AzureSpeechSynthesizer:
    """
    Wrap Azure Speech SDK synthesis for WAV output.
    """

    def __init__(
        self,
        credentials: AzureSpeechCredentials,
        voice_name: str = "en-US-JennyNeural",
    ) -> None:
        self.credentials = credentials
        self.voice_name = voice_name

    def synthesize(self, text: str, detection: EmotionDetectionResult) -> AzureSpeechSynthesisResult:
        speechsdk = self._load_sdk()
        ssml = build_emotional_ssml(text, detection, voice_name=self.voice_name)

        speech_config = speechsdk.SpeechConfig(
            subscription=self.credentials.key,
            region=self.credentials.region,
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.CancellationDetails(result)
            raise RuntimeError(
                "Azure Speech synthesis failed. "
                f"Reason: {details.reason}. "
                f"Error details: {details.error_details or 'No extra details provided.'}"
            )

        return AzureSpeechSynthesisResult(
            wav_bytes=result.audio_data,
            voice_name=self.voice_name,
            style=get_style_preset(detection.category, detection.confidence).style,
            ssml=ssml,
        )

    @staticmethod
    def _load_sdk():
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise RuntimeError(
                "Azure Speech SDK is not installed. "
                "Install it with: python -m pip install azure-cognitiveservices-speech"
            ) from exc
        return speechsdk
