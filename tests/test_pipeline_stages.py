from pathlib import Path
from uuid import uuid4

import numpy as np

from tts.config import TTSConfig
from tts.emotion.emotion_labels import EmotionCategory, EmotionLabel, EmotionVector
from tts.pipeline import EmotionTTSPipeline
from tts.text.ssml_parser import AnnotatedSegment


TEST_TEMP_ROOT = Path(__file__).resolve().parent / "_tmp"


def make_temp_path(filename: str) -> Path:
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_TEMP_ROOT / f"{uuid4().hex}-{filename}"


class DummyAcousticModel:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, emotion=None):
        self.calls.append((text, emotion))
        return np.ones((4, 80), dtype=np.float32)


class DummyVocoder:
    def __init__(self):
        self.calls = []
        self.saves = []

    def synthesize(self, mel, sample_rate):
        self.calls.append((np.array(mel, copy=True), sample_rate))
        return np.linspace(-0.1, 0.1, 8, dtype=np.float32)

    def save(self, waveform, path, sample_rate):
        self.saves.append((np.array(waveform, copy=True), path, sample_rate))
        with open(path, "wb") as handle:
            handle.write(b"test")


class DummyEmotionAnalyzer:
    def analyze(self, text):
        return EmotionLabel(
            category=EmotionCategory.HAPPY,
            vector=EmotionVector(0.8, 0.7, 0.6, 0.9),
            confidence=0.9,
        )

    def analyze_segments(self, segments):
        return [self.analyze(segment) for segment in segments]


class FailEmotionAnalyzer:
    def analyze(self, text):
        raise AssertionError("emotion analyzer should not have been used")

    def analyze_segments(self, segments):
        raise AssertionError("emotion analyzer should not have been used")


class DummyTextProcessor:
    def process(self, text, level="sentence"):
        return ["first segment", "second segment"]


class DummySSMLParser:
    def parse(self, ssml):
        return [
            AnnotatedSegment(text="hello there", emotion_category=EmotionCategory.HAPPY),
            AnnotatedSegment(text="", emotion_category=EmotionCategory.NEUTRAL),
            AnnotatedSegment(text="plain text"),
        ]


def test_text_to_mel_saves_intermediate_file():
    pipeline = EmotionTTSPipeline(TTSConfig(device="cpu"))
    pipeline._acoustic_model = DummyAcousticModel()
    pipeline._emotion_analyzer = DummyEmotionAnalyzer()

    output_path = make_temp_path("mel.npy")
    mel = pipeline.text_to_mel("Hello world", emotion="happy", output_path=output_path)

    assert output_path.exists()
    np.testing.assert_array_equal(np.load(output_path), mel)
    assert pipeline._acoustic_model.calls[0][1] is not None


def test_mel_to_audio_accepts_saved_mel_file():
    pipeline = EmotionTTSPipeline(TTSConfig(device="cpu"))
    pipeline._vocoder = DummyVocoder()

    mel_path = make_temp_path("mel.npy")
    audio_path = make_temp_path("audio.wav")
    mel = np.ones((3, 80), dtype=np.float32)
    np.save(mel_path, mel)

    waveform = pipeline.mel_to_audio(mel_path, output_path=audio_path)

    assert audio_path.exists()
    np.testing.assert_array_equal(pipeline._vocoder.calls[0][0], mel)
    assert waveform.shape == (8,)


def test_text_to_mel_skips_emotion_analysis_when_conditioning_disabled():
    pipeline = EmotionTTSPipeline(TTSConfig(device="cpu", enable_emotion_conditioning=False))
    pipeline._acoustic_model = DummyAcousticModel()
    pipeline._emotion_analyzer = FailEmotionAnalyzer()

    mel = pipeline.text_to_mel("Hello world", emotion="happy")

    assert mel.shape == (4, 80)
    assert pipeline._acoustic_model.calls[0][1] is None


def test_long_form_skips_emotion_analysis_when_conditioning_disabled():
    pipeline = EmotionTTSPipeline(TTSConfig(device="cpu", enable_emotion_conditioning=False))
    pipeline._text_processor = DummyTextProcessor()
    pipeline._acoustic_model = DummyAcousticModel()
    pipeline._vocoder = DummyVocoder()
    pipeline._emotion_analyzer = FailEmotionAnalyzer()

    waveform = pipeline.synthesize_long_form("first. second.")

    assert waveform.size > 0
    assert all(emotion is None for _, emotion in pipeline._acoustic_model.calls)


def test_ssml_skips_emotion_analysis_when_conditioning_disabled():
    pipeline = EmotionTTSPipeline(TTSConfig(device="cpu", enable_emotion_conditioning=False))
    pipeline._ssml_parser = DummySSMLParser()
    pipeline._acoustic_model = DummyAcousticModel()
    pipeline._vocoder = DummyVocoder()
    pipeline._emotion_analyzer = FailEmotionAnalyzer()

    waveform = pipeline.synthesize_ssml("<speak>test</speak>")

    assert waveform.size > 0
    assert all(emotion is None for _, emotion in pipeline._acoustic_model.calls)
