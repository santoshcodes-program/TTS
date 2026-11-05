from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts.config import TTSConfig
from tts.vocoder.vocoder import HiFiGANVocoder


class MelToAudioSynthesizer:
    """Standalone helper for the mel -> audio stage."""

    def __init__(self, config: TTSConfig | None = None, device: str | None = None) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or TTSConfig(device=resolved_device)
        self.config.device = resolved_device
        self.vocoder = HiFiGANVocoder(
            checkpoint_path=self.config.vocoder_checkpoint_path,
            device=self.config.device,
        )

    def mel_to_audio(
        self,
        mel: Union[str, Path, np.ndarray],
        output_path: str = "output.wav",
    ) -> np.ndarray:
        if isinstance(mel, (str, Path)):
            mel_array = np.load(Path(mel))
        else:
            mel_array = np.asarray(mel)

        waveform = self.vocoder.synthesize(mel_array, self.config.sample_rate)
        self.vocoder.save(waveform, output_path, self.config.sample_rate)
        return waveform


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio from a mel spectrogram.")
    parser.add_argument("--mel", default="mel.npy", help="Path to the input mel spectrogram (.npy).")
    parser.add_argument("--output", default="output.wav", help="Path to save the generated waveform.")
    parser.add_argument("--device", default=None, help="Inference device, for example 'cpu' or 'cuda'.")
    args = parser.parse_args()

    synthesizer = MelToAudioSynthesizer(device=args.device)
    waveform = synthesizer.mel_to_audio(args.mel, output_path=args.output)
    print(f"Audio generated: {args.output} ({len(waveform)} samples)")


if __name__ == "__main__":
    main()
