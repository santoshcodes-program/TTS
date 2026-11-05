"""Tests for the EmotionCurve module."""

import numpy as np
import pytest

from tts.emotion.emotion_labels import EmotionVector
from tts.emotion.emotion_curve import EmotionCurve, EmotionKeyframe


class TestEmotionKeyframe:
    """Tests for EmotionKeyframe."""

    def test_creation(self):
        kf = EmotionKeyframe(0.5, EmotionVector(0.8, 0.6, 0.5, 0.7))
        assert kf.time_position == 0.5
        assert kf.emotion.valence == 0.8

    def test_time_clamping(self):
        kf = EmotionKeyframe(1.5, EmotionVector())
        assert kf.time_position == 1.0

        kf2 = EmotionKeyframe(-0.5, EmotionVector())
        assert kf2.time_position == 0.0


class TestEmotionCurve:
    """Tests for EmotionCurve."""

    def _make_two_keyframes(self):
        return [
            EmotionKeyframe(0.0, EmotionVector(0.0, 0.0, 0.0, 0.0)),
            EmotionKeyframe(1.0, EmotionVector(1.0, 1.0, 1.0, 1.0)),
        ]

    def _make_three_keyframes(self):
        return [
            EmotionKeyframe(0.0, EmotionVector(0.0, 0.0, 0.5, 0.0)),
            EmotionKeyframe(0.5, EmotionVector(0.9, 0.8, 0.5, 0.9)),
            EmotionKeyframe(1.0, EmotionVector(0.2, 0.3, 0.5, 0.3)),
        ]

    def test_single_keyframe(self):
        kf = EmotionKeyframe(0.5, EmotionVector(0.7, 0.3, 0.5, 0.6))
        curve = EmotionCurve([kf])
        sampled = curve.sample(0.0)
        assert sampled.valence == pytest.approx(0.7)
        sampled2 = curve.sample(1.0)
        assert sampled2.valence == pytest.approx(0.7)

    def test_linear_interpolation_two_points(self):
        curve = EmotionCurve(self._make_two_keyframes())
        mid = curve.sample(0.5)
        assert mid.valence == pytest.approx(0.5, abs=0.05)
        assert mid.arousal == pytest.approx(0.5, abs=0.05)

    def test_endpoints_match(self):
        curve = EmotionCurve(self._make_two_keyframes())
        start = curve.sample(0.0)
        end = curve.sample(1.0)
        assert start.valence == pytest.approx(0.0, abs=0.01)
        assert end.valence == pytest.approx(1.0, abs=0.01)

    def test_cubic_spline_three_points(self):
        curve = EmotionCurve(self._make_three_keyframes())
        mid = curve.sample(0.5)
        # At t=0.5 the keyframe is (0.9, 0.8), so should be close
        assert mid.valence == pytest.approx(0.9, abs=0.05)

    def test_continuity(self):
        """Curve should be continuous — no big jumps between close samples."""
        curve = EmotionCurve(self._make_three_keyframes())
        prev = curve.sample(0.0)
        for i in range(1, 100):
            t = i / 100.0
            curr = curve.sample(t)
            jump = abs(curr.valence - prev.valence)
            assert jump < 0.15, f"Discontinuity at t={t}: jump={jump}"
            prev = curr

    def test_from_segments(self):
        emotions = [
            EmotionVector(0.2, 0.3, 0.5, 0.5),
            EmotionVector(0.8, 0.7, 0.5, 0.8),
        ]
        curve = EmotionCurve.from_segments(emotions)
        assert len(curve) == 2

    def test_constant_curve(self):
        ev = EmotionVector(0.5, 0.5, 0.5, 0.5)
        curve = EmotionCurve.constant(ev)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = curve.sample(t)
            assert s.valence == pytest.approx(0.5, abs=0.01)

    def test_scale_intensity(self):
        curve = EmotionCurve(self._make_two_keyframes())
        scaled = curve.scale_intensity(0.5)
        end = scaled.sample(1.0)
        assert end.intensity == pytest.approx(0.5, abs=0.05)

    def test_to_frame_sequence_shape(self):
        curve = EmotionCurve(self._make_three_keyframes())
        frames = curve.to_frame_sequence(100)
        assert frames.shape == (100, 4)

    def test_to_frame_sequence_single_frame(self):
        curve = EmotionCurve(self._make_two_keyframes())
        frames = curve.to_frame_sequence(1)
        assert frames.shape == (1, 4)

    def test_to_frame_sequence_zero_frames(self):
        curve = EmotionCurve(self._make_two_keyframes())
        frames = curve.to_frame_sequence(0)
        assert frames.shape == (0, 4)

    def test_empty_keyframes_raises(self):
        with pytest.raises(ValueError):
            EmotionCurve([])

    def test_len(self):
        curve = EmotionCurve(self._make_three_keyframes())
        assert len(curve) == 3
