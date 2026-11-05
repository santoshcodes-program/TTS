from tts.emotion.emotion_labels import EmotionCategory
from tts.emotion.simple_emotion_detector import EmotionDetectionResult
from tts.providers.azure_speech import build_emotional_ssml, get_style_preset


def test_happy_maps_to_cheerful_style():
    preset = get_style_preset(EmotionCategory.HAPPY, confidence=0.8)

    assert preset.style == "cheerful"
    assert preset.style_degree > 1.0


def test_fearful_maps_to_terrified_style():
    preset = get_style_preset(EmotionCategory.FEARFUL, confidence=0.9)

    assert preset.style == "terrified"


def test_ssml_contains_voice_and_prosody():
    detection = EmotionDetectionResult(
        category=EmotionCategory.SAD,
        confidence=0.7,
        scores={"sad": 0.7},
        source="heuristic",
        evidence="sad words",
    )

    ssml = build_emotional_ssml(
        "I am feeling very sad today.",
        detection,
        voice_name="en-US-JennyNeural",
    )

    assert 'voice name="en-US-JennyNeural"' in ssml
    assert 'style="sad"' in ssml
    assert "<prosody" in ssml
