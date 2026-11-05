from tts.emotion.emotion_labels import EmotionCategory
from tts.emotion.simple_emotion_detector import EmotionDetector


def test_detect_happy_text():
    detector = EmotionDetector(prefer_transformers=False)
    result = detector.detect("I am so happy and excited to celebrate this wonderful news!")

    assert result.category == EmotionCategory.HAPPY
    assert result.confidence > 0.2
    assert result.source == "heuristic"


def test_detect_sad_text():
    detector = EmotionDetector(prefer_transformers=False)
    result = detector.detect("I miss you and I feel lonely and heartbroken tonight.")

    assert result.category == EmotionCategory.SAD
    assert result.confidence > 0.2


def test_detect_neutral_text():
    detector = EmotionDetector(prefer_transformers=False)
    result = detector.detect("The meeting starts at nine in the morning.")

    assert result.category == EmotionCategory.NEUTRAL
    assert result.scores["neutral"] >= 0.2
