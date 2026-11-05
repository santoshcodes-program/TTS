"""Tests for text processing and SSML parsing."""

import pytest
from tts.text.text_processor import TextProcessor
from tts.text.ssml_parser import SSMLParser, AnnotatedSegment
from tts.emotion.emotion_labels import EmotionCategory


class TestTextProcessor:
    """Tests for TextProcessor."""

    def setup_method(self):
        self.tp = TextProcessor()

    # ── Sentence segmentation ────────────────────────────────────

    def test_single_sentence(self):
        result = self.tp.segment_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        text = "Hello world. How are you? I'm fine!"
        result = self.tp.segment_sentences(text)
        assert len(result) == 3

    def test_empty_text(self):
        assert self.tp.segment_sentences("") == []
        assert self.tp.segment_sentences("   ") == []

    def test_abbreviation_protection(self):
        text = "Dr. Smith went to St. Louis."
        result = self.tp.segment_sentences(text)
        # Should NOT split on "Dr." or "St."
        assert len(result) == 1

    # ── Clause segmentation ──────────────────────────────────────

    def test_clause_split_comma(self):
        text = "I went home, and then I ate"
        result = self.tp.segment_clauses(text)
        assert len(result) >= 2

    def test_clause_split_semicolon(self):
        text = "First thing; second thing"
        result = self.tp.segment_clauses(text)
        assert len(result) == 2

    def test_single_clause(self):
        text = "Just one clause"
        result = self.tp.segment_clauses(text)
        assert result == ["Just one clause"]

    # ── Normalization ────────────────────────────────────────────

    def test_normalize_abbreviations(self):
        result = self.tp.normalize("Dr. Smith met Mr. Jones")
        assert "doctor" in result
        assert "mister" in result

    def test_normalize_numbers(self):
        result = self.tp.normalize("I have 3 cats and 5 dogs")
        assert "three" in result
        assert "five" in result

    def test_normalize_whitespace(self):
        result = self.tp.normalize("  too   many   spaces  ")
        assert "  " not in result

    # ── Process convenience ──────────────────────────────────────

    def test_process_sentence_level(self):
        text = "Hello. World."
        result = self.tp.process(text, level="sentence")
        assert len(result) == 2

    def test_process_clause_level(self):
        text = "Hello, world. Nice day, isn't it?"
        result = self.tp.process(text, level="clause")
        assert len(result) >= 3


class TestSSMLParser:
    """Tests for SSMLParser."""

    def setup_method(self):
        self.parser = SSMLParser()

    def test_plain_text(self):
        result = self.parser.parse("Hello world")
        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].has_emotion is False

    def test_empty_text(self):
        assert self.parser.parse("") == []
        assert self.parser.parse("   ") == []

    def test_single_emotion_tag(self):
        ssml = '<emotion category="happy" intensity="0.8">Great day!</emotion>'
        result = self.parser.parse(ssml)
        assert len(result) >= 1
        seg = result[0]
        assert seg.text == "Great day!"
        assert seg.emotion_category == EmotionCategory.HAPPY
        assert seg.emotion_intensity == pytest.approx(0.8)

    def test_multiple_emotion_tags(self):
        ssml = """
        <speak>
            <emotion category="happy" intensity="0.9">What a wonderful day!</emotion>
            <emotion category="sad" intensity="0.6">But I miss you.</emotion>
        </speak>
        """
        result = self.parser.parse(ssml)
        emotions = [seg for seg in result if seg.has_emotion]
        assert len(emotions) == 2

    def test_mixed_plain_and_emotion(self):
        ssml = """
        <speak>
            Plain text here.
            <emotion category="angry">I am furious!</emotion>
        </speak>
        """
        result = self.parser.parse(ssml)
        assert len(result) >= 2

    def test_malformed_xml_fallback(self):
        bad_xml = "<emotion>unclosed tag"
        result = self.parser.parse(bad_xml)
        assert len(result) >= 1  # should not crash

    def test_emotion_vector_override(self):
        ssml = '<emotion category="happy" valence="0.9" arousal="0.8">Nice!</emotion>'
        result = self.parser.parse(ssml)
        seg = result[0]
        assert seg.emotion_override is not None
        assert seg.emotion_override.valence == pytest.approx(0.9)
