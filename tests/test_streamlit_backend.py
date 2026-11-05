from io import BytesIO
import wave

import numpy as np

from tts.emotion.emotion_labels import EmotionCategory, EmotionLabel, EmotionVector
from tts.streamlit_backend import (
    mel_to_display_image,
    resolve_emotion_for_preview,
    synthesize_emotional_preview,
    synthesize_preview,
    waveform_to_wav_bytes,
)


class DummyEmotionAnalyzer:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def analyze(self, text):
        self.calls.append(text)
        return self.label


class DummyPipeline:
    def __init__(self, label=None, threshold=0.4):
        self.config = type(
            "Config",
            (),
            {
                "sample_rate": 22050,
                "emotion_confidence_threshold": threshold,
            },
        )()
        self.calls = []
        self.emotion_analyzer = DummyEmotionAnalyzer(
            label
            or EmotionLabel(
                category=EmotionCategory.HAPPY,
                vector=EmotionVector(0.8, 0.7, 0.6, 0.9),
                confidence=0.9,
            )
        )

    def text_to_mel(self, text, emotion=None, intensity=1.0):
        self.calls.append(("text_to_mel", text, emotion, intensity))
        return np.ones((6, 80), dtype=np.float32)

    def mel_to_audio(self, mel):
        self.calls.append(("mel_to_audio", tuple(mel.shape)))
        return np.linspace(-0.5, 0.5, 16, dtype=np.float32)


def test_synthesize_preview_uses_staged_pipeline():
    pipeline = DummyPipeline()

    artifacts = synthesize_preview(
        pipeline,
        text="hello there",
        emotion="happy",
        intensity=0.75,
    )

    assert artifacts.sample_rate == 22050
    assert artifacts.mel.shape == (6, 80)
    assert artifacts.waveform.shape == (16,)
    assert pipeline.calls[0] == ("text_to_mel", "hello there", "happy", 0.75)
    assert pipeline.calls[1] == ("mel_to_audio", (6, 80))


def test_resolve_emotion_for_preview_manual_mode():
    pipeline = DummyPipeline()

    resolved = resolve_emotion_for_preview(
        pipeline,
        text="hello there",
        emotion_mode="manual",
        manual_emotion="sad",
        intensity=0.65,
    )

    assert resolved.emotion_input == "sad"
    assert resolved.intensity == 0.65
    assert resolved.display_name == "sad"
    assert resolved.source == "manual"


def test_synthesize_emotional_preview_auto_mode_uses_detected_vector():
    pipeline = DummyPipeline()

    artifacts = synthesize_emotional_preview(
        pipeline,
        text="hello there",
        emotion_mode="auto",
        intensity=0.5,
    )

    text_call = pipeline.calls[0]
    assert text_call[0] == "text_to_mel"
    assert isinstance(text_call[2], EmotionVector)
    assert text_call[2].intensity == 0.45
    assert text_call[3] == 1.0
    assert artifacts.emotion_name == "happy"
    assert artifacts.emotion_source == "auto"
    assert artifacts.emotion_confidence == 0.9


def test_synthesize_emotional_preview_low_confidence_falls_back_to_neutral():
    pipeline = DummyPipeline(
        label=EmotionLabel(
            category=EmotionCategory.ANGRY,
            vector=EmotionVector(0.2, 0.9, 0.8, 0.2),
            confidence=0.2,
        )
    )

    artifacts = synthesize_emotional_preview(
        pipeline,
        text="hello there",
        emotion_mode="auto",
        intensity=1.0,
    )

    assert pipeline.calls[0] == ("text_to_mel", "hello there", "neutral", 1.0)
    assert artifacts.emotion_name == "neutral"
    assert artifacts.emotion_source == "auto-low-confidence"
    assert artifacts.emotion_confidence == 0.2


def test_waveform_to_wav_bytes_roundtrip():
    waveform = np.linspace(-0.2, 0.2, 32, dtype=np.float32)
    wav_bytes = waveform_to_wav_bytes(waveform, 22050)

    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 22050
        assert wav_file.getnchannels() == 1
        frames = wav_file.readframes(wav_file.getnframes())

    restored = np.frombuffer(frames, dtype=np.int16)
    assert restored.shape == waveform.shape


def test_mel_to_display_image_returns_rgb_heatmap():
    mel = np.arange(80 * 12, dtype=np.float32).reshape(12, 80)
    image = mel_to_display_image(mel)

    assert image.shape == (80, 12, 3)
    assert image.dtype == np.uint8
    assert image.max() <= 255
    assert image.min() >= 0
