from tts.emotion.emotion_labels import EmotionCategory
from tts.providers.edge_speech import get_edge_preset


def test_happy_edge_preset_has_positive_pitch_and_rate():
    preset = get_edge_preset(EmotionCategory.HAPPY, confidence=0.8)

    assert preset.voice == "en-US-JennyNeural"
    assert preset.rate.startswith("+")
    assert preset.pitch.startswith("+")


def test_sad_edge_preset_has_lower_rate_and_pitch():
    preset = get_edge_preset(EmotionCategory.SAD, confidence=0.9)

    assert preset.rate.startswith("-")
    assert preset.pitch.startswith("-")


def test_neutral_edge_preset_is_flat():
    preset = get_edge_preset(EmotionCategory.NEUTRAL, confidence=1.0)

    assert preset.rate == "+0%"
    assert preset.pitch == "+0Hz"
