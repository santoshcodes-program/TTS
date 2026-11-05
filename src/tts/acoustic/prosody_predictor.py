"""
Emotion-aware prosody predictor.

Generates pitch and energy contour modifiers based on emotion vectors,
using psychologically-motivated mappings from the VAD space to
acoustic prosody parameters.
"""

from __future__ import annotations

import numpy as np

from tts.emotion.emotion_labels import EmotionVector


class ProsodyPredictor:
    """
    Predicts pitch and energy contours as modification curves
    that can be applied to mel spectrograms.

    The predictor generates smooth contours using sinusoidal components
    modulated by emotion parameters, simulating natural prosodic
    variation associated with different emotional states.
    """

    def __init__(self, base_pitch_hz: float = 150.0) -> None:
        """
        Parameters
        ----------
        base_pitch_hz : float
            Reference pitch in Hz (used for relative offset calculation).
        """
        self.base_pitch_hz = base_pitch_hz

    def predict_pitch_contour(
        self,
        emotion: EmotionVector,
        n_frames: int,
    ) -> np.ndarray:
        """
        Generate a pitch offset contour based on emotion.

        Returns
        -------
        np.ndarray
            Pitch offset in semitones for each frame. Shape (n_frames,).
            Positive = higher pitch, negative = lower.
        """
        if n_frames <= 0:
            return np.array([])

        t = np.linspace(0, 1, n_frames)

        # Base offset from arousal
        base_offset = (emotion.arousal - 0.5) * 4.0  # ±2 semitones

        # Valence modulation
        if emotion.valence > 0.6:
            # Happy: rising intonation pattern
            pattern = 0.5 * np.sin(2 * np.pi * t * 2.0)  # gentle wave
        elif emotion.valence < 0.3:
            if emotion.arousal > 0.6:
                # Angry: sharp, erratic pitch
                pattern = 1.0 * np.sin(2 * np.pi * t * 5.0) * np.random.uniform(0.7, 1.3, n_frames)
            else:
                # Sad: gradually declining pitch
                pattern = -0.5 * t
        else:
            pattern = np.zeros(n_frames)

        # Fear: trembling pitch
        if emotion.valence < 0.3 and emotion.arousal > 0.7:
            tremor = 0.3 * np.sin(2 * np.pi * t * 8.0)
            pattern = pattern + tremor

        contour = (base_offset + pattern) * emotion.intensity
        return contour.astype(np.float32)

    def predict_energy_contour(
        self,
        emotion: EmotionVector,
        n_frames: int,
    ) -> np.ndarray:
        """
        Generate an energy multiplier contour based on emotion.

        Returns
        -------
        np.ndarray
            Energy multiplier for each frame. Shape (n_frames,).
            1.0 = no change, >1.0 = louder, <1.0 = softer.
        """
        if n_frames <= 0:
            return np.array([])

        t = np.linspace(0, 1, n_frames)

        # Base energy from arousal and dominance
        base = 1.0 + (emotion.arousal - 0.5) * 0.6
        base += (emotion.dominance - 0.5) * 0.2

        # Variation pattern
        if emotion.arousal > 0.7:
            # High arousal: more dynamic range
            variation = 0.15 * np.sin(2 * np.pi * t * 3.0)
        elif emotion.arousal < 0.3:
            # Low arousal: compressed, subdued
            variation = -0.1 * np.ones(n_frames)
        else:
            variation = np.zeros(n_frames)

        contour = (base + variation * emotion.intensity)
        return np.clip(contour, 0.3, 2.0).astype(np.float32)

    def predict_speaking_rate(self, emotion: EmotionVector) -> float:
        """
        Predict a global speaking rate modifier.

        Returns
        -------
        float
            Rate multiplier: >1.0 = faster, <1.0 = slower.
        """
        # High arousal → faster; sad (low V, low A) → slower
        rate = 1.0 + (emotion.arousal - 0.5) * 0.4 * emotion.intensity

        if emotion.valence < 0.3 and emotion.arousal < 0.4:
            rate -= 0.15 * emotion.intensity  # sad slowdown

        return max(0.6, min(1.5, rate))
