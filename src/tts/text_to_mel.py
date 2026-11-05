from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.config import TTSConfig
from tts.pipeline import EmotionTTSPipeline


class FastSpeech2TTS:
    """
    Backwards-compatible helper for the text -> mel stage.

    This delegates to the main pipeline so the standalone stage and the
    end-to-end synthesis path stay aligned.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_tag: Optional[str] = None,
        config: Optional[TTSConfig] = None,
    ) -> None:
        self.config = config or TTSConfig(device=device)
        self.config.device = device
        if model_tag is not None:
            self.config.fastspeech2_model_tag = model_tag
        self.pipeline = EmotionTTSPipeline(self.config)

    def text_to_mel(
        self,
        text: str,
        save_path: Optional[str] = "mel.npy",
        emotion: Optional[str] = None,
        intensity: float = 1.0,
    ) -> np.ndarray:
        return self.pipeline.text_to_mel(
            text,
            emotion=emotion,
            intensity=intensity,
            output_path=save_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mel spectrogram from text.")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello, welcome to my emotion based text to speech system.",
    )
    parser.add_argument("--output", default="mel.npy", help="Path to save the mel spectrogram (.npy).")
    parser.add_argument("--device", default="cpu", help="Inference device, for example 'cpu' or 'cuda'.")
    parser.add_argument("--emotion", default=None, help="Optional emotion override such as 'happy'.")
    parser.add_argument("--intensity", type=float, default=1.0, help="Emotion intensity multiplier.")
    args = parser.parse_args()

    tts = FastSpeech2TTS(device=args.device)
    mel = tts.text_to_mel(
        args.text,
        save_path=args.output,
        emotion=args.emotion,
        intensity=args.intensity,
    )
    print("Mel shape:", mel.shape)


if __name__ == "__main__":
    main()
