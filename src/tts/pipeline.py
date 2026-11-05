"""
End-to-end emotion-aware TTS pipeline.

Orchestrates text processing, emotion analysis, acoustic synthesis,
and vocoding through one consistent text -> mel -> audio flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from tts.config import TTSConfig
from tts.emotion.emotion_curve import EmotionCurve
from tts.emotion.emotion_labels import EmotionCategory, EmotionVector, get_default_vector
from tts.text.ssml_parser import SSMLParser
from tts.text.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class EmotionTTSPipeline:
    """
    Main entry point for emotion-aware text-to-speech synthesis.

    Supports:
    1. Simple synthesis with an optional emotion override
    2. Long-form synthesis with smooth transitions across segments
    3. SSML-like markup with explicit per-segment emotion control
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        self.config.setup_logging()

        self._text_processor = TextProcessor()
        self._ssml_parser = SSMLParser()
        self._emotion_analyzer = None
        self._acoustic_model = None
        self._vocoder = None

        logger.info("EmotionTTSPipeline initialized (device=%s)", self.config.device)

    @property
    def emotion_analyzer(self):
        if self._emotion_analyzer is None:
            from tts.emotion.text_emotion_analyzer import TextEmotionAnalyzer

            self._emotion_analyzer = TextEmotionAnalyzer(
                model_name=self.config.emotion_model_name,
                device=self.config.device,
                confidence_threshold=self.config.emotion_confidence_threshold,
            )
        return self._emotion_analyzer

    @property
    def acoustic_model(self):
        if self._acoustic_model is None:
            from tts.acoustic.fastspeech2_model import EmotionAwareFastSpeech2

            self._acoustic_model = EmotionAwareFastSpeech2(
                model_tag=self.config.fastspeech2_model_tag,
                device=self.config.device,
                pitch_range=self.config.pitch_range,
                energy_range=self.config.energy_range,
                rate_range=self.config.rate_range,
                smoothing_window=self.config.smoothing_window,
            )
        return self._acoustic_model

    @property
    def vocoder(self):
        if self._vocoder is None:
            from tts.vocoder.vocoder import HiFiGANVocoder

            self._vocoder = HiFiGANVocoder(
                checkpoint_path=self.config.vocoder_checkpoint_path,
                device=self.config.device,
            )
        return self._vocoder

    def text_to_mel(
        self,
        text: str,
        emotion: Optional[Union[str, EmotionCategory, EmotionVector]] = None,
        intensity: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Convert text into a mel spectrogram.

        This exposes the first half of the synthesis pipeline directly so the
        intermediate mel can be inspected, saved, or passed to a separate step.
        """
        if self.config.enable_emotion_conditioning:
            emotion_vector = self._resolve_emotion(text, emotion, intensity)
            if emotion_vector is not None:
                mel = self.acoustic_model.synthesize(text, emotion=emotion_vector)
            else:
                mel = self.acoustic_model.synthesize(text)
        else:
            if emotion is not None:
                logger.debug("Emotion conditioning is disabled; ignoring emotion override.")
            mel = self.acoustic_model.synthesize(text)

        if output_path:
            mel_path = Path(output_path)
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, mel)
            logger.info("Mel spectrogram saved to %s (shape: %s)", mel_path, mel.shape)

        return mel

    def mel_to_audio(
        self,
        mel: Union[np.ndarray, str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Convert a mel spectrogram array or `.npy` file into audio.
        """
        if isinstance(mel, (str, Path)):
            mel_array = np.load(Path(mel))
        else:
            mel_array = np.asarray(mel)

        waveform = self.vocoder.synthesize(mel_array, self.config.sample_rate)
        if output_path:
            self.vocoder.save(waveform, str(output_path), self.config.sample_rate)
        return waveform

    def synthesize(
        self,
        text: str,
        emotion: Optional[Union[str, EmotionCategory, EmotionVector]] = None,
        intensity: float = 1.0,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text with optional emotion control.
        """
        logger.info("Synthesizing: '%s' (emotion=%s, intensity=%.2f)", text[:50], emotion, intensity)
        mel = self.text_to_mel(text, emotion=emotion, intensity=intensity)
        return self.mel_to_audio(mel, output_path=output_path)

    def synthesize_long_form(
        self,
        text: str,
        output_path: Optional[str] = None,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesize long-form text with optional smooth emotion transitions.
        """
        logger.info("Long-form synthesis: %d characters", len(text))

        segments = self._text_processor.process(text, level=self.config.segmentation_level)
        if not segments:
            segments = [text]

        logger.info("Segmented into %d parts", len(segments))

        emotion_labels = None
        curve = None
        if self.config.enable_emotion_conditioning:
            emotion_labels = self.emotion_analyzer.analyze_segments(segments)
            curve = EmotionCurve.from_segments([label.vector for label in emotion_labels])
            if intensity != 1.0:
                curve = curve.scale_intensity(intensity)

        waveforms: List[np.ndarray] = []
        for index, segment in enumerate(segments):
            emotion_at_t = None
            if curve is not None and emotion_labels is not None:
                label = emotion_labels[index]
                logger.info(
                    "  Segment %d/%d: [%s] '%s'",
                    index + 1,
                    len(segments),
                    label.category.value,
                    segment[:40],
                )
                emotion_at_t = curve.sample((index + 0.5) / len(segments))
            else:
                logger.info("  Segment %d/%d: '%s'", index + 1, len(segments), segment[:40])

            if emotion_at_t is not None:
                mel = self.acoustic_model.synthesize(segment, emotion=emotion_at_t)
            else:
                mel = self.acoustic_model.synthesize(segment)

            waveforms.append(self.mel_to_audio(mel))

            if index < len(segments) - 1:
                silence = np.zeros(int(self.config.sample_rate * 0.15), dtype=np.float32)
                waveforms.append(silence)

        combined = np.concatenate(waveforms) if waveforms else np.array([], dtype=np.float32)
        if output_path:
            self.vocoder.save(combined, str(output_path), self.config.sample_rate)
        return combined

    def synthesize_ssml(
        self,
        ssml: str,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from SSML-like markup with optional emotion tags.
        """
        logger.info("SSML synthesis")

        annotated_segments = self._ssml_parser.parse(ssml)
        waveforms: List[np.ndarray] = []

        for segment in annotated_segments:
            if not segment.text.strip():
                waveforms.append(np.zeros(int(self.config.sample_rate * 0.3), dtype=np.float32))
                continue

            emotion_vector = None
            if self.config.enable_emotion_conditioning:
                if segment.has_emotion:
                    emotion_vector = segment.get_vector()
                else:
                    emotion_vector = self.emotion_analyzer.analyze(segment.text).vector

            if emotion_vector is not None:
                mel = self.acoustic_model.synthesize(segment.text, emotion=emotion_vector)
            else:
                mel = self.acoustic_model.synthesize(segment.text)

            waveforms.append(self.mel_to_audio(mel))

        combined = np.concatenate(waveforms) if waveforms else np.array([], dtype=np.float32)
        if output_path:
            self.vocoder.save(combined, str(output_path), self.config.sample_rate)
        return combined

    def _resolve_emotion(
        self,
        text: str,
        emotion: Optional[Union[str, EmotionCategory, EmotionVector]],
        intensity: float,
    ) -> Optional[EmotionVector]:
        """Resolve the emotion parameter into an EmotionVector."""
        if emotion is None:
            label = self.emotion_analyzer.analyze(text)
            return label.vector.scaled(intensity) if label.is_reliable() else None

        if isinstance(emotion, EmotionVector):
            return emotion.scaled(intensity)

        if isinstance(emotion, str):
            category = EmotionCategory.from_string(emotion)
        elif isinstance(emotion, EmotionCategory):
            category = emotion
        else:
            raise TypeError(f"Invalid emotion type: {type(emotion)}")

        return get_default_vector(category, intensity)
