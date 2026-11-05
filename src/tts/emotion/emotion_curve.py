"""
Emotion curves: continuous, time-varying emotion trajectories.

Given a set of keyframe emotions (e.g. from per-sentence analysis),
this module builds a smooth curve via cubic spline interpolation
so emotion changes gradually rather than jumping abruptly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tts.emotion.emotion_labels import EmotionVector


@dataclass
class EmotionKeyframe:
    """
    A single point on the emotion timeline.

    Parameters
    ----------
    time_position : float
        Normalized time position in [0.0, 1.0] where 0.0 is the start
        of the utterance and 1.0 is the end.
    emotion : EmotionVector
        The emotion state at this time position.
    """

    time_position: float
    emotion: EmotionVector

    def __post_init__(self) -> None:
        self.time_position = max(0.0, min(1.0, self.time_position))


class EmotionCurve:
    """
    A time-varying emotion trajectory built from keyframes.

    Supports:
    - Cubic spline interpolation for smooth transitions
    - Linear interpolation fallback for 2 keyframes
    - Global intensity scaling
    - Discretization to frame-level arrays for acoustic conditioning
    """

    def __init__(self, keyframes: List[EmotionKeyframe]) -> None:
        if not keyframes:
            raise ValueError("At least one keyframe is required.")

        # Sort by time
        self._keyframes = sorted(keyframes, key=lambda k: k.time_position)

        # Pre-compute arrays for interpolation
        self._times = np.array([k.time_position for k in self._keyframes])
        self._values = np.array([k.emotion.to_list() for k in self._keyframes])
        # shape: (n_keyframes, 4) — [valence, arousal, dominance, intensity]

        self._splines: Optional[list] = None
        self._build_interpolators()

    # ── Construction helpers ─────────────────────────────────────────

    @classmethod
    def from_keyframes(cls, keyframes: List[EmotionKeyframe]) -> "EmotionCurve":
        """Build a curve from explicit keyframes."""
        return cls(keyframes)

    @classmethod
    def from_segments(
        cls,
        emotions: List[EmotionVector],
        durations: Optional[List[float]] = None,
    ) -> "EmotionCurve":
        """
        Build a curve from per-segment emotion vectors.

        Each segment gets a keyframe placed at its midpoint.
        If *durations* are not provided, segments are assumed equal length.
        """
        n = len(emotions)
        if n == 0:
            raise ValueError("At least one emotion segment is required.")

        if durations is None:
            durations = [1.0 / n] * n

        # Normalize durations to sum to 1.0
        total = sum(durations)
        durations = [d / total for d in durations]

        # Place keyframes at segment midpoints
        keyframes: List[EmotionKeyframe] = []
        cumulative = 0.0
        for emotion, dur in zip(emotions, durations):
            midpoint = cumulative + dur / 2.0
            keyframes.append(EmotionKeyframe(time_position=midpoint, emotion=emotion))
            cumulative += dur

        return cls(keyframes)

    @classmethod
    def constant(cls, emotion: EmotionVector) -> "EmotionCurve":
        """Build a flat curve that holds the same emotion throughout."""
        return cls([
            EmotionKeyframe(time_position=0.0, emotion=emotion),
            EmotionKeyframe(time_position=1.0, emotion=emotion),
        ])

    # ── Interpolation ────────────────────────────────────────────────

    def _build_interpolators(self) -> None:
        """Build cubic spline interpolators for each VAD+intensity channel."""
        from scipy.interpolate import CubicSpline

        n = len(self._keyframes)

        if n == 1:
            # Single keyframe → constant curve
            self._splines = None
            return

        if n == 2:
            # Two keyframes → linear interpolation (no cubic needed)
            self._splines = None
            return

        # n >= 3 → cubic spline per channel
        self._splines = []
        for ch in range(4):  # valence, arousal, dominance, intensity
            spline = CubicSpline(
                self._times,
                self._values[:, ch],
                bc_type="clamped",  # zero derivative at endpoints
            )
            self._splines.append(spline)

    def sample(self, t: float) -> EmotionVector:
        """
        Sample the emotion curve at normalized time *t* ∈ [0, 1].

        Returns an interpolated EmotionVector.
        """
        t = max(0.0, min(1.0, t))
        n = len(self._keyframes)

        if n == 1:
            return self._keyframes[0].emotion

        if self._splines is not None:
            # Cubic spline
            vals = [float(spline(t)) for spline in self._splines]
        else:
            # Linear interpolation between 2 points
            vals = []
            for ch in range(4):
                v0, v1 = self._values[0, ch], self._values[-1, ch]
                t0, t1 = self._times[0], self._times[-1]
                if t1 == t0:
                    vals.append(float(v0))
                else:
                    ratio = (t - t0) / (t1 - t0)
                    vals.append(float(v0 + (v1 - v0) * ratio))

        return EmotionVector(
            valence=vals[0],
            arousal=vals[1],
            dominance=vals[2],
            intensity=vals[3],
        )

    # ── Utilities ────────────────────────────────────────────────────

    def scale_intensity(self, factor: float) -> "EmotionCurve":
        """
        Return a new curve with all keyframe intensities scaled by *factor*.
        """
        new_keyframes = [
            EmotionKeyframe(
                time_position=kf.time_position,
                emotion=kf.emotion.scaled(factor),
            )
            for kf in self._keyframes
        ]
        return EmotionCurve(new_keyframes)

    def to_frame_sequence(self, n_frames: int) -> np.ndarray:
        """
        Discretize the curve into an array of shape (n_frames, 4).

        Each row is [valence, arousal, dominance, intensity] sampled
        at evenly spaced time positions.
        """
        if n_frames <= 0:
            return np.empty((0, 4))

        if n_frames == 1:
            return np.array([self.sample(0.5).to_list()])

        times = np.linspace(0.0, 1.0, n_frames)
        frames = np.array([self.sample(float(t)).to_list() for t in times])
        return frames

    @property
    def keyframes(self) -> List[EmotionKeyframe]:
        """Return a copy of the keyframes."""
        return list(self._keyframes)

    def __len__(self) -> int:
        return len(self._keyframes)

    def __repr__(self) -> str:
        return f"EmotionCurve(n_keyframes={len(self._keyframes)})"
