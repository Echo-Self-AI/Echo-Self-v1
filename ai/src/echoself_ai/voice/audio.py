"""WAV audio file I/O, format validation, resampling, and normalization."""
import base64
from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
import struct
from typing import Any, Optional, Union
import wave

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Validation Error Taxonomy
# -----------------------------------------------------------------------------

class MediaValidationError(ValueError):
    """Base exception for all media validation failures."""
    pass


class AudioValidationError(MediaValidationError):
    """Base exception for audio-specific validation failures."""
    pass


class InvalidAudioFormatError(AudioValidationError):
    """Raised when an audio file has an unsupported container, header, or bit-depth."""
    pass


class AudioDurationError(AudioValidationError):
    """Raised when an audio recording is shorter or longer than allowable limits."""
    pass


class AudioFileSizeError(AudioValidationError):
    """Raised when an audio file exceeds safety size thresholds."""
    pass


class CorruptAudioError(AudioValidationError):
    """Raised when an audio file is truncated, contains NaNs, or has malformed data chunks."""
    pass


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Standardized audio processing configuration."""

    sample_rate: int = 16000
    max_duration_seconds: float = 60.0
    min_duration_seconds: float = 0.1
    max_file_size_bytes: int = 25 * 1024 * 1024  # 25 MB safety ceiling
    normalize_amplitude: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate < 8000:
            raise ValueError(f"sample_rate must be at least 8000 Hz, got {self.sample_rate}")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be greater than min_duration_seconds")


# -----------------------------------------------------------------------------
# Resampling
# -----------------------------------------------------------------------------

def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Resample 1D waveform tensor to target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return waveform

    orig_dim = waveform.dim()
    if orig_dim == 1:
        x = waveform.view(1, 1, -1)
    elif orig_dim == 2:
        x = waveform.unsqueeze(1)
    else:
        raise ValueError(f"Expected 1D or 2D waveform tensor, got shape {waveform.shape}")

    new_len = max(1, int(round(x.shape[-1] * (target_sr / orig_sr))))
    new_x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)

    if orig_dim == 1:
        return new_x.squeeze(0).squeeze(0)
    return new_x.squeeze(1)


# -----------------------------------------------------------------------------
# Validation & Ingestion
# -----------------------------------------------------------------------------

def ingest_audio(
    source: Union[str, Path, bytes, BytesIO],
    config: Optional[AudioConfig] = None,
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int]:
    """Ingest, validate, downmix, and standardize an input WAV audio recording.

    Accepts file paths, raw bytes, or BytesIO buffers. Validates headers, bit depth,
    duration, file size, and numerical finiteness.

    Returns:
        waveform: 1D float32 PyTorch tensor in [-1.0, 1.0] resampled to target_sr.
        sample_rate: Output sample rate (equal to target_sr).
    """
    cfg = config or AudioConfig(sample_rate=target_sr)

    # Resolve byte buffer or file
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {p}")
        file_size = p.stat().st_size
        if file_size > cfg.max_file_size_bytes:
            raise AudioFileSizeError(
                f"Audio file size ({file_size} bytes) exceeds safety limit {cfg.max_file_size_bytes} bytes"
            )
        if file_size < 44:  # Smallest valid RIFF WAV header is 44 bytes
            raise CorruptAudioError(f"Audio file too small ({file_size} bytes) to contain a valid WAV header")
        bio = open(p, "rb")
        needs_close = True
    elif isinstance(source, bytes):
        if len(source) > cfg.max_file_size_bytes:
            raise AudioFileSizeError(
                f"Audio byte buffer ({len(source)} bytes) exceeds safety limit {cfg.max_file_size_bytes} bytes"
            )
        if len(source) < 44:
            raise CorruptAudioError(f"Audio buffer too small ({len(source)} bytes) to contain a valid WAV header")
        bio = BytesIO(source)
        needs_close = False
    elif isinstance(source, BytesIO):
        source.seek(0, 2)
        size = source.tell()
        source.seek(0)
        if size > cfg.max_file_size_bytes:
            raise AudioFileSizeError(
                f"Audio stream ({size} bytes) exceeds safety limit {cfg.max_file_size_bytes} bytes"
            )
        if size < 44:
            raise CorruptAudioError(f"Audio stream too small ({size} bytes) to contain a valid WAV header")
        bio = source
        needs_close = False
    else:
        raise InvalidAudioFormatError(f"Unsupported audio source type: {type(source).__name__}")

    try:
        try:
            wf = wave.open(bio, "rb")
        except wave.Error as exc:
            raise InvalidAudioFormatError(f"Malformed or non-WAV audio container: {exc}") from exc
        except (EOFError, struct.error) as exc:
            raise CorruptAudioError(f"Corrupt WAV chunk structure: {exc}") from exc

        with wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            in_sr = wf.getframerate()
            n_frames = wf.getnframes()

            if n_channels < 1:
                raise CorruptAudioError("Audio reports 0 channels")

            if sampwidth != 2:
                raise InvalidAudioFormatError(
                    f"Unsupported bit depth: expected 16-bit Linear PCM (sampwidth=2), got sampwidth={sampwidth}"
                )

            if in_sr < 8000 or in_sr > 96000:
                raise InvalidAudioFormatError(
                    f"Sample rate {in_sr} Hz outside supported range [8000, 96000] Hz"
                )

            if n_frames == 0:
                raise CorruptAudioError("Audio file contains 0 audio frames")

            raw_bytes = wf.readframes(n_frames)
            expected_bytes = n_frames * n_channels * sampwidth
            if len(raw_bytes) != expected_bytes:
                raise CorruptAudioError(
                    f"Truncated WAV file: expected {expected_bytes} bytes, received {len(raw_bytes)}"
                )

        num_samples = n_frames * n_channels
        fmt = f"<{num_samples}h"
        try:
            ints = struct.unpack(fmt, raw_bytes)
        except struct.error as exc:
            raise CorruptAudioError(f"Failed to unpack PCM sample bytes: {exc}") from exc

        tensor = torch.tensor(ints, dtype=torch.float32) / 32768.0

        if not torch.isfinite(tensor).all():
            raise CorruptAudioError("Audio contains non-finite (NaN or Inf) sample values")

        # Downmix multichannel to mono
        if n_channels > 1:
            tensor = tensor.view(-1, n_channels).mean(dim=1)

        raw_duration = len(tensor) / float(in_sr)
        if raw_duration < cfg.min_duration_seconds:
            raise AudioDurationError(
                f"Audio duration {raw_duration:.2f}s is less than minimum allowable {cfg.min_duration_seconds}s"
            )
        if raw_duration > cfg.max_duration_seconds:
            raise AudioDurationError(
                f"Audio duration {raw_duration:.2f}s exceeds maximum allowable {cfg.max_duration_seconds}s"
            )

        # Resample to target sample rate if needed
        if in_sr != target_sr:
            tensor = resample_waveform(tensor, orig_sr=in_sr, target_sr=target_sr)

        if cfg.normalize_amplitude:
            tensor = normalize_waveform(tensor)

        return tensor, target_sr

    finally:
        if needs_close:
            bio.close()


def load_wav(path: Union[str, Path], config: Optional[AudioConfig] = None) -> tuple[torch.Tensor, int]:
    """Backwards-compatible wrapper calling ingest_audio."""
    if config is None:
        cfg = AudioConfig(min_duration_seconds=0.001, normalize_amplitude=False)
    else:
        cfg = config
    return ingest_audio(path, config=cfg, target_sr=cfg.sample_rate)



def write_wav(path: Union[str, Path, Any], waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    """Save a 1D float32 PyTorch tensor as a 16-bit PCM WAV file or file-like object."""
    waveform = waveform.detach().cpu().squeeze()
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform tensor, got shape {waveform.shape}")

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
