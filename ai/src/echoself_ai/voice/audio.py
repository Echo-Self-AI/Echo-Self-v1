from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any, Optional, Union
import wave

import torch


@dataclass
class AudioConfig:
    """Standardized audio processing configuration."""

    sample_rate: int = 16000
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.1
    normalize_amplitude: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate < 8000:
            raise ValueError(f"sample_rate must be at least 8000 Hz, got {self.sample_rate}")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be greater than min_duration_seconds")


def load_wav(path: str | Path, config: Optional[AudioConfig] = None) -> tuple[torch.Tensor, int]:
    """Load a 16-bit PCM WAV file into a normalized 1D float32 PyTorch tensor.

    Converts multichannel audio to mono by averaging channels.

    Returns:
        waveform: Tensor of shape (num_samples,) with values in [-1.0, 1.0].
        sample_rate: Detected sampling rate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        if sampwidth != 2:
            raise ValueError(
                f"Unsupported bit depth: expected 16-bit PCM (sampwidth=2), got sampwidth={sampwidth}"
            )

        raw_bytes = wf.readframes(n_frames)

    # 16-bit signed integer unpacking
    num_samples = n_frames * n_channels
    fmt = f"<{num_samples}h"
    ints = struct.unpack(fmt, raw_bytes)

    # Convert to float32 in [-1.0, 1.0]
    tensor = torch.tensor(ints, dtype=torch.float32) / 32768.0

    if n_channels > 1:
        # Reshape to (n_frames, n_channels) and average across channels to obtain mono
        tensor = tensor.view(-1, n_channels).mean(dim=1)

    duration = len(tensor) / sample_rate
    if config is not None:
        if duration < config.min_duration_seconds:
            raise ValueError(
                f"Audio duration {duration:.2f}s is less than minimum {config.min_duration_seconds}s"
            )
        if duration > config.max_duration_seconds:
            raise ValueError(
                f"Audio duration {duration:.2f}s exceeds maximum {config.max_duration_seconds}s"
            )
        if config.normalize_amplitude:
            tensor = normalize_waveform(tensor)

    return tensor, sample_rate


def write_wav(path: Union[str, Path, Any], waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a 1D float32 PyTorch tensor as a 16-bit PCM WAV file or file-like object."""
    # Ensure 1D and on CPU
    waveform = waveform.detach().cpu().squeeze()
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform tensor, got shape {waveform.shape}")

    # Clamp to prevent overflow wrap-around distortion
    clamped = torch.clamp(waveform, -1.0, 1.0)
    ints = (clamped * 32767.0).to(torch.int16).tolist()

    raw_bytes = struct.pack(f"<{len(ints)}h", *ints)

    if isinstance(path, (str, Path)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        target = str(p)
    else:
        target = path

    with wave.open(target, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)


save_wav = write_wav



def normalize_waveform(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Peak-normalize a waveform tensor to avoid digital clipping."""
    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        return (waveform / peak) * target_peak
    return waveform


def generate_synthetic_tone(
    freq_hz: float = 440.0,
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.8,
) -> torch.Tensor:
    """Generate a synthetic pure sinusoidal waveform for deterministic testing."""
    num_samples = int(duration_seconds * sample_rate)
    t = torch.arange(0, num_samples, dtype=torch.float32) / sample_rate
    waveform = amplitude * torch.sin(2.0 * math.pi * freq_hz * t)
    return waveform

