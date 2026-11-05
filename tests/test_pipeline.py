"""
Integration tests for the EmotionTTSPipeline.

These tests require ML models (ESPnet, HiFi-GAN, HuggingFace)
to be available and are marked with @pytest.mark.slow.
"""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from tts.config import TTSConfig
from tts.emotion.emotion_labels import EmotionCategory, EmotionVector


TEST_TEMP_ROOT = Path(__file__).resolve().parent / "_tmp"


def make_temp_path(filename: str) -> Path:
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return TEST_TEMP_ROOT / f"{uuid4().hex}-{filename}"


class TestTTSConfig:
    """Tests for TTSConfig."""

    def test_default_config(self):
        cfg = TTSConfig()
        assert cfg.device == "cpu"
        assert cfg.sample_rate == 22050
        assert cfg.enable_emotion_conditioning is True

    def test_default_emotion_category(self):
        cfg = TTSConfig(default_emotion="happy")
        assert cfg.default_emotion_category == EmotionCategory.HAPPY

    def test_invalid_segmentation_level(self):
        with pytest.raises(AssertionError):
            TTSConfig(segmentation_level="word")

    def test_yaml_roundtrip(self):
        cfg = TTSConfig(device="cuda", sample_rate=44100)
        yaml_path = str(make_temp_path("config.yml"))
        cfg.to_yaml(yaml_path)

        loaded = TTSConfig.from_yaml(yaml_path)
        assert loaded.device == "cuda"
        assert loaded.sample_rate == 44100


@pytest.mark.slow
class TestPipelineIntegration:
    """
    End-to-end pipeline tests. Marked slow since they require
    downloading and loading large ML models.

    Run with: pytest -m slow
    """

    @pytest.fixture
    def pipeline(self):
        from tts.pipeline import EmotionTTSPipeline

        cfg = TTSConfig(device="cpu")
        return EmotionTTSPipeline(cfg)

    def test_synthesize_simple(self, pipeline):
        output = make_temp_path("test.wav")
        wav = pipeline.synthesize(
            "Hello world.",
            emotion="neutral",
            output_path=str(output),
        )
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0
        assert output.exists()

    def test_synthesize_with_emotion(self, pipeline):
        wav = pipeline.synthesize(
            "I am so happy today!",
            emotion="happy",
            intensity=0.9,
        )
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0

    def test_synthesize_long_form(self, pipeline):
        text = (
            "Once upon a time there was a happy prince. "
            "But one day something terrible happened. "
            "He became very sad and lonely."
        )
        output = make_temp_path("longform.wav")
        wav = pipeline.synthesize_long_form(text, output_path=str(output))
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0

    def test_synthesize_ssml(self, pipeline):
        ssml = """
        <speak>
            <emotion category="happy" intensity="0.9">
                What a wonderful day!
            </emotion>
            <emotion category="sad" intensity="0.6">
                But I miss you so much.
            </emotion>
        </speak>
        """
        output = make_temp_path("ssml.wav")
        wav = pipeline.synthesize_ssml(ssml, output_path=str(output))
        assert isinstance(wav, np.ndarray)
        assert len(wav) > 0
