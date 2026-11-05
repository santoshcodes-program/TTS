"""
Text processing utilities for the emotion-aware TTS system.

Handles sentence splitting, clause segmentation, and basic text
normalization to prepare input for both emotion analysis and
acoustic synthesis.
"""

from __future__ import annotations

import re
from typing import List


class TextProcessor:
    """
    Segments and normalizes text for the TTS pipeline.

    Provides hierarchical segmentation:
    - Sentence-level (for coarse emotion assignment)
    - Clause-level (for finer emotion granularity)
    """

    # Sentence boundary pattern: split on .!? followed by space or end
    _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

    # Clause boundary pattern: split on , ; : — and conjunctions
    _CLAUSE_RE = re.compile(
        r'(?<=[,;:])\s+|'           # after punctuation
        r'\s+(?:but|and|or|yet|so|because|although|while|when|if)\s+',
        re.IGNORECASE,
    )

    # Number-to-word mappings for basic normalization
    _NUMBER_WORDS = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve",
    }

    # Common abbreviations
    _ABBREVIATIONS = {
        "mr.": "mister",
        "mrs.": "missus",
        "ms.": "miss",
        "dr.": "doctor",
        "prof.": "professor",
        "st.": "saint",
        "jr.": "junior",
        "sr.": "senior",
        "vs.": "versus",
        "etc.": "etcetera",
        "govt.": "government",
    }

    def segment_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences on .!? boundaries.

        Returns a list of non-empty, stripped sentences.
        """
        if not text or not text.strip():
            return []

        # Protect abbreviations from being split (case-insensitive)
        protected = text
        for abbr in self._ABBREVIATIONS:
            # Match both lowercase and title-case versions
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
            titled = abbr[0].upper() + abbr[1:]
            protected = protected.replace(titled, titled.replace(".", "<DOT>"))

        parts = self._SENTENCE_RE.split(protected)

        # Restore dots
        sentences = [
            s.replace("<DOT>", ".").strip()
            for s in parts
            if s.strip()
        ]
        return sentences

    def segment_clauses(self, text: str) -> List[str]:
        """
        Split text into clauses for finer-grained emotion analysis.

        Splits on commas, semicolons, colons, and coordinating/
        subordinating conjunctions.
        """
        if not text or not text.strip():
            return []

        parts = self._CLAUSE_RE.split(text)
        clauses = [c.strip() for c in parts if c and c.strip()]
        return clauses if clauses else [text.strip()]

    def normalize(self, text: str) -> str:
        """
        Basic text normalization:
        - Expand common abbreviations
        - Convert small numbers to words
        - Collapse whitespace
        - Strip leading/trailing whitespace
        """
        result = text.lower()

        # Expand abbreviations
        for abbr, expansion in self._ABBREVIATIONS.items():
            result = result.replace(abbr, expansion)

        # Convert standalone small numbers to words
        def _replace_number(match: re.Match) -> str:
            num = match.group(0)
            return self._NUMBER_WORDS.get(num, num)

        result = re.sub(r'\b(\d{1,2})\b', _replace_number, result)

        # Collapse whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def process(self, text: str, level: str = "sentence") -> List[str]:
        """
        Convenience method: normalize and then segment.

        Parameters
        ----------
        text : str
            The input text to process.
        level : str
            Segmentation granularity: "sentence" or "clause".
        """
        normalized = self.normalize(text)

        if level == "clause":
            # First split into sentences, then each sentence into clauses
            sentences = self.segment_sentences(normalized)
            clauses = []
            for sentence in sentences:
                clauses.extend(self.segment_clauses(sentence))
            return clauses
        else:
            return self.segment_sentences(normalized)
