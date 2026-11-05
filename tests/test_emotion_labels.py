"""Tests for emotion labels, vectors, and mappings."""

import pytest
from tts.emotion.emotion_labels import (
    EmotionCategory,
    EmotionVector,
    EmotionLabel,
    DEFAULT_VAD_MAP,
    get_default_vector,
)


class TestEmotionCategory:
    """Tests for EmotionCategory enum."""

    def test_all_categories_exist(self):
        expected = {"neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"}
        actual = {e.value for e in EmotionCategory}
        assert actual == expected

    def test_from_string_by_value(self):
        assert EmotionCategory.from_string("happy") == EmotionCategory.HAPPY
        assert EmotionCategory.from_string("HAPPY") == EmotionCategory.HAPPY

    def test_from_string_by_name(self):
        assert EmotionCategory.from_string("ANGRY") == EmotionCategory.ANGRY

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown emotion"):
            EmotionCategory.from_string("ecstatic")

    def test_from_string_whitespace(self):
        assert EmotionCategory.from_string("  sad  ") == EmotionCategory.SAD


class TestEmotionVector:
    """Tests for EmotionVector dataclass."""

    def test_default_values(self):
        v = EmotionVector()
        assert v.valence == 0.5
        assert v.arousal == 0.5
        assert v.dominance == 0.5
        assert v.intensity == 0.5

    def test_clamping_high(self):
        v = EmotionVector(valence=1.5, arousal=2.0, dominance=3.0, intensity=10.0)
        assert v.valence == 1.0
        assert v.arousal == 1.0
        assert v.dominance == 1.0
        assert v.intensity == 1.0

    def test_clamping_low(self):
        v = EmotionVector(valence=-0.5, arousal=-1.0, dominance=-2.0, intensity=-5.0)
        assert v.valence == 0.0
        assert v.arousal == 0.0
        assert v.dominance == 0.0
        assert v.intensity == 0.0

    def test_to_list(self):
        v = EmotionVector(0.1, 0.2, 0.3, 0.4)
        assert v.to_list() == [0.1, 0.2, 0.3, 0.4]

    def test_scaled(self):
        v = EmotionVector(0.8, 0.6, 0.7, 0.8)
        scaled = v.scaled(0.5)
        assert scaled.intensity == pytest.approx(0.4)
        assert scaled.valence == 0.8  # valence unchanged

    def test_neutral(self):
        n = EmotionVector.neutral()
        assert n.intensity == 0.0
        assert n.valence == 0.5

    def test_lerp(self):
        a = EmotionVector(0.0, 0.0, 0.0, 0.0)
        b = EmotionVector(1.0, 1.0, 1.0, 1.0)
        mid = EmotionVector.lerp(a, b, 0.5)
        assert mid.valence == pytest.approx(0.5)
        assert mid.arousal == pytest.approx(0.5)

    def test_lerp_clamped_t(self):
        a = EmotionVector(0.0, 0.0, 0.0, 0.0)
        b = EmotionVector(1.0, 1.0, 1.0, 1.0)
        result = EmotionVector.lerp(a, b, 2.0)  # t > 1
        assert result.valence == pytest.approx(1.0)


class TestEmotionLabel:
    """Tests for EmotionLabel dataclass."""

    def test_creation(self):
        label = EmotionLabel(
            category=EmotionCategory.HAPPY,
            vector=EmotionVector(0.85, 0.7, 0.65, 0.8),
            confidence=0.92,
        )
        assert label.category == EmotionCategory.HAPPY
        assert label.confidence == 0.92

    def test_confidence_clamping(self):
        label = EmotionLabel(
            category=EmotionCategory.SAD,
            vector=EmotionVector(),
            confidence=1.5,
        )
        assert label.confidence == 1.0

    def test_is_reliable_above_threshold(self):
        label = EmotionLabel(EmotionCategory.ANGRY, EmotionVector(), 0.8)
        assert label.is_reliable(threshold=0.4) is True

    def test_is_reliable_below_threshold(self):
        label = EmotionLabel(EmotionCategory.ANGRY, EmotionVector(), 0.2)
        assert label.is_reliable(threshold=0.4) is False


class TestDefaultVADMap:
    """Tests for the default category → VAD mapping."""

    def test_all_categories_mapped(self):
        for cat in EmotionCategory:
            assert cat in DEFAULT_VAD_MAP

    def test_happy_high_valence(self):
        happy = DEFAULT_VAD_MAP[EmotionCategory.HAPPY]
        assert happy.valence > 0.7

    def test_sad_low_valence(self):
        sad = DEFAULT_VAD_MAP[EmotionCategory.SAD]
        assert sad.valence < 0.3

    def test_angry_high_arousal(self):
        angry = DEFAULT_VAD_MAP[EmotionCategory.ANGRY]
        assert angry.arousal > 0.7

    def test_get_default_vector(self):
        v = get_default_vector(EmotionCategory.HAPPY)
        assert v.valence == pytest.approx(0.85)

    def test_get_default_vector_scaled(self):
        v = get_default_vector(EmotionCategory.HAPPY, intensity=0.5)
        assert v.intensity == pytest.approx(0.4)
