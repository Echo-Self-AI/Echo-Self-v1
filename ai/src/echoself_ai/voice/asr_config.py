"""ASR configuration dataclass for the EchoSelf speech recognition module."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ASRConfig:
    """Configuration for the character-level CTC ASR model.

    Attributes:
        vocab_size: Total number of output classes including the blank token.
            For character-level ASCII ASR this is typically 29
            (26 letters + space + apostrophe + blank).
        blank_id: Index of the CTC blank token.  Convention: last index = vocab_size - 1.
        n_mels: Number of mel-spectrogram channels fed to the encoder.
        n_fft: FFT size for log-mel extraction.
        hop_length: STFT hop length in samples.
        win_length: STFT window length in samples.
        sample_rate: Expected waveform sample rate.
        cnn_channels: Number of output channels in the CNN front-end.
        cnn_kernel_size: Kernel size for the 2-D CNN front-end.
        hidden_dim: Hidden dimension for GRU encoder layers.
        num_layers: Number of bidirectional GRU layers.
        dropout: Dropout probability applied between encoder layers.
        checkpoint_path: Optional path to a saved model checkpoint.
            When set, SpeechRecognizer will attempt to load it on
            construction.
    """

    vocab_size: int = 29  # 26 chars + space + apostrophe + blank
    blank_id: int = 28    # last index by convention
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    sample_rate: int = 16000
    cnn_channels: int = 32
    cnn_kernel_size: int = 3
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2 (one class + blank)")
        if self.blank_id < 0 or self.blank_id >= self.vocab_size:
            raise ValueError(
                f"blank_id {self.blank_id} out of range [0, {self.vocab_size})"
            )
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.n_mels < 1:
            raise ValueError("n_mels must be positive")
        if self.sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000")
