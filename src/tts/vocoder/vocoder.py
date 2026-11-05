"""
HiFi-GAN vocoder wrapper for mel-to-waveform synthesis.

Provides a clean class-based API around the ParallelWaveGAN/HiFi-GAN
checkpoint for converting mel spectrograms into audio waveforms.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HiFiGANVocoder:
    """
    Neural vocoder that converts mel spectrograms to audio waveforms.
    """

    def __init__(
        self,
        checkpoint_path: str = "hifigan.v1/checkpoint-400000steps.pkl",
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load the vocoder model if not already loaded."""
        if self._model is not None:
            return

        try:
            load_model = self._import_load_model()
            checkpoint_path = self._resolve_checkpoint_path()

            logger.info("Loading HiFi-GAN vocoder: %s", checkpoint_path)
            device = torch.device(self.device)
            self._model = load_model(str(checkpoint_path)).to(device)
            self._model.remove_weight_norm()
            self._model.eval()
            logger.info("HiFi-GAN vocoder loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "parallel_wavegan is not installed and the bundled copy could not be loaded."
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Vocoder checkpoint not found: {self.checkpoint_path}. "
                f"Make sure the checkpoint file exists."
            )
        except Exception as exc:
            logger.error("Failed to load vocoder: %s", exc)
            raise RuntimeError(f"Could not load vocoder: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize(
        self,
        mel: np.ndarray,
        sample_rate: int = 22050,
    ) -> np.ndarray:
        """
        Convert a mel spectrogram to a waveform.
        """
        self._ensure_loaded()

        del sample_rate  # metadata only, kept for API compatibility

        device = torch.device(self.device)
        mel_array = np.asarray(mel, dtype=np.float32)

        if mel_array.ndim == 2 and mel_array.shape[0] == 80 and mel_array.shape[1] != 80:
            mel_array = mel_array.T

        if mel_array.ndim == 2:
            mel_tensor = torch.from_numpy(mel_array).unsqueeze(0).to(device)
        elif mel_array.ndim == 3:
            mel_tensor = torch.from_numpy(mel_array).to(device)
        else:
            raise ValueError(f"Expected mel with 2 or 3 dimensions, got {mel_array.ndim}")

        with torch.no_grad():
            waveform = self._model.inference(mel_tensor).view(-1).cpu().numpy()

        return waveform

    def save(
        self,
        waveform: np.ndarray,
        path: str,
        sample_rate: int = 22050,
    ) -> None:
        """
        Save a waveform to a WAV file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        from tts.streamlit_backend import write_wave_file

        write_wave_file(str(output_path), waveform, sample_rate)
        logger.info("Audio saved to %s (%.2f seconds)", output_path, len(waveform) / sample_rate)

    def mel_to_audio(
        self,
        mel: np.ndarray,
        output_path: str,
        sample_rate: int = 22050,
    ) -> np.ndarray:
        """
        Convenience helper that synthesizes audio from mel and saves it.
        """
        waveform = self.synthesize(mel, sample_rate)
        self.save(waveform, output_path, sample_rate)
        return waveform

    def _resolve_checkpoint_path(self) -> Path:
        checkpoint_path = Path(self.checkpoint_path).expanduser()
        if checkpoint_path.is_absolute():
            return checkpoint_path

        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            Path.cwd() / checkpoint_path,
            project_root / checkpoint_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[-1]

    @staticmethod
    def _import_load_model():
        try:
            from parallel_wavegan.utils import load_model

            return load_model
        except ImportError:
            bundled_root = Path(__file__).resolve().parents[3] / "ParallelWaveGAN"
            if not bundled_root.exists():
                raise

            if str(bundled_root) not in sys.path:
                sys.path.insert(0, str(bundled_root))

            for module_name in list(sys.modules):
                if module_name == "parallel_wavegan" or module_name.startswith("parallel_wavegan."):
                    sys.modules.pop(module_name, None)

            from parallel_wavegan.utils import load_model

            return load_model
