"""
Emotion conditioning for mel spectrograms.

Applies post-hoc prosodic modifications (pitch, energy, duration)
to mel spectrograms based on VAD emotion vectors. This approach
works WITHOUT retraining the base FastSpeech2 model — it modulates
the output mel to express different emotional characteristics.

Acoustic correlates of emotion (based on speech science literature):
    Happy   → higher pitch, higher energy, faster rate
    Sad     → lower pitch, lower energy, slower rate
    Angry   → higher pitch, much higher energy, faster rate, wider pitch range
    Fearful → higher pitch, variable energy, faster rate
    Surprised → pitch spikes, higher energy
    Disgusted → lower pitch, compressed energy
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from tts.emotion.emotion_labels import EmotionVector

logger = logging.getLogger(__name__)


class EmotionConditioner:
    """
    Modifies mel spectrograms to express emotional prosody.

    Uses the VAD (Valence-Arousal-Dominance) + intensity dimensions
    to compute pitch shift, energy scaling, and speech rate factors.
    """

    def __init__(
        self,
        pitch_range: float = 3.0,
        energy_range: float = 0.5,
        rate_range: float = 0.3,
        smoothing_window: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        pitch_range : float
            Maximum pitch shift in semitones (positive or negative).
        energy_range : float
            Maximum energy scaling factor offset (e.g. ±0.5 → 0.5× to 1.5×).
        rate_range : float
            Maximum speech rate factor offset (e.g. ±0.3 → 0.7× to 1.3×).
        smoothing_window : int
            Window size for smoothing per-frame modifications.
        """
        self.pitch_range = pitch_range
        self.energy_range = energy_range
        self.rate_range = rate_range
        self.smoothing_window = smoothing_window

    # ── Static conditioning (uniform emotion) ────────────────────────

    def condition_mel_static(
        self,
        mel: np.ndarray,
        emotion: EmotionVector,
    ) -> np.ndarray:
        """
        Apply uniform emotion conditioning across the entire mel.

        Parameters
        ----------
        mel : np.ndarray
            Mel spectrogram of shape (T, n_mels).
        emotion : EmotionVector
            The target emotion.

        Returns
        -------
        np.ndarray
            Modified mel spectrogram.
        """
        if emotion.intensity < 0.01:
            return mel  # No modification for neutral/zero intensity

        mel = mel.copy()
        mel = self._apply_pitch_shift(mel, emotion)
        mel = self._apply_energy_scaling(mel, emotion)
        return mel

    # ── Dynamic conditioning (per-frame emotion curve) ───────────────

    def condition_mel_dynamic(
        self,
        mel: np.ndarray,
        emotion_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Apply time-varying emotion conditioning frame by frame.

        Parameters
        ----------
        mel : np.ndarray
            Mel spectrogram of shape (T, n_mels).
        emotion_frames : np.ndarray
            Emotion parameters of shape (N, 4) where columns are
            [valence, arousal, dominance, intensity].
            Will be resampled to T frames if N != T.

        Returns
        -------
        np.ndarray
            Modified mel spectrogram.
        """
        T, n_mels = mel.shape
        mel = mel.copy()

        # Resample emotion frames to match mel length
        if emotion_frames.shape[0] != T:
            emotion_frames = self._resample_frames(emotion_frames, T)

        # Compute per-frame modification factors
        for t in range(T):
            ev = EmotionVector(
                valence=float(emotion_frames[t, 0]),
                arousal=float(emotion_frames[t, 1]),
                dominance=float(emotion_frames[t, 2]),
                intensity=float(emotion_frames[t, 3]),
            )

            if ev.intensity < 0.01:
                continue

            # Pitch: shift mel frequency bins
            pitch_shift = self._compute_pitch_factor(ev)
            mel[t, :] = self._shift_mel_bins(mel[t, :], pitch_shift)

            # Energy: scale amplitude
            energy_factor = self._compute_energy_factor(ev)
            mel[t, :] *= energy_factor

        # Smooth the result to avoid frame-to-frame artifacts
        if self.smoothing_window > 1 and T > self.smoothing_window:
            mel = uniform_filter1d(mel, size=self.smoothing_window, axis=0)

        return mel

    # ── Prosody computation ──────────────────────────────────────────

    def _compute_pitch_factor(self, ev: EmotionVector) -> float:
        """
        Compute pitch shift in semitones from emotion vector.

        High arousal + high valence → raise pitch (happy, excited)
        High arousal + low valence → raise pitch (angry, fearful)
        Low arousal + low valence → lower pitch (sad)
        """
        # Arousal is the primary driver of pitch change
        arousal_offset = (ev.arousal - 0.5) * 2.0  # [-1, 1]

        # Valence modulates direction slightly
        valence_mod = (ev.valence - 0.5) * 0.5  # [-0.25, 0.25]

        raw = (arousal_offset + valence_mod) * self.pitch_range
        return raw * ev.intensity

    def _compute_energy_factor(self, ev: EmotionVector) -> float:
        """
        Compute energy multiplier from emotion vector.

        High arousal → louder; Low arousal → softer.
        Dominance adds a small boost.
        """
        arousal_factor = (ev.arousal - 0.5) * 2.0 * self.energy_range
        dominance_factor = (ev.dominance - 0.5) * 0.5 * self.energy_range
        raw = 1.0 + (arousal_factor + dominance_factor) * ev.intensity
        return max(0.3, min(2.0, raw))  # clamp to safe range

    def _compute_rate_factor(self, ev: EmotionVector) -> float:
        """
        Compute speech rate factor from emotion vector.

        High arousal → faster; Low valence + low arousal → slower (sad).
        """
        arousal_factor = (ev.arousal - 0.5) * 2.0 * self.rate_range
        # Sad speech is slower: low valence + low arousal
        if ev.valence < 0.3 and ev.arousal < 0.4:
            arousal_factor -= 0.15
        raw = 1.0 + arousal_factor * ev.intensity
        return max(0.6, min(1.5, raw))

    # ── Mel manipulation helpers ─────────────────────────────────────

    def _apply_pitch_shift(self, mel: np.ndarray, ev: EmotionVector) -> np.ndarray:
        """Apply pitch shift to the full mel spectrogram."""
        shift = self._compute_pitch_factor(ev)
        if abs(shift) < 0.05:
            return mel

        T, n_mels = mel.shape
        for t in range(T):
            mel[t, :] = self._shift_mel_bins(mel[t, :], shift)
        return mel

    def _apply_energy_scaling(self, mel: np.ndarray, ev: EmotionVector) -> np.ndarray:
        """Apply energy scaling to the full mel spectrogram."""
        factor = self._compute_energy_factor(ev)
        if abs(factor - 1.0) < 0.01:
            return mel
        return mel * factor

    @staticmethod
    def _shift_mel_bins(frame: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift mel-scale frequency bins by a number of semitones.

        This simulates a pitch shift by shifting the spectral envelope
        up or down along the mel axis.
        """
        n = len(frame)
        if abs(semitones) < 0.05:
            return frame

        # Convert semitones to bin shift (approximate: 1 semitone ≈ 1 mel bin)
        bin_shift = semitones * 0.8  # scale factor for mel resolution

        shifted = np.zeros_like(frame)
        for i in range(n):
            src = i - bin_shift
            src_int = int(np.floor(src))
            frac = src - src_int

            if 0 <= src_int < n - 1:
                shifted[i] = frame[src_int] * (1.0 - frac) + frame[src_int + 1] * frac
            elif src_int == n - 1:
                shifted[i] = frame[src_int]
            # else: zero (out of bounds)

        return shifted

    @staticmethod
    def _resample_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
        """Resample emotion frames to a target length via linear interpolation."""
        src_len = frames.shape[0]
        if src_len == target_len:
            return frames

        indices = np.linspace(0, src_len - 1, target_len)
        resampled = np.zeros((target_len, frames.shape[1]))
        for ch in range(frames.shape[1]):
            resampled[:, ch] = np.interp(indices, np.arange(src_len), frames[:, ch])
        return resampled
