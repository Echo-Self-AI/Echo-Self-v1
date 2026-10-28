import math
from typing import Optional

import torch
from torch import nn


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Create a triangular Mel-filterbank matrix using PyTorch primitives.

    Converts linear STFT frequency bins to perceptually spaced Mel bins.

    Returns:
        Tensor of shape (n_mels, 1 + n_fft // 2).
    """
    if f_max is None:
        f_max = float(sample_rate // 2)

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Linearly spaced points in Mel scale
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])

    # Convert Hz points to FFT bin indices
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    num_bins = 1 + n_fft // 2
    filterbank = torch.zeros(n_mels, num_bins, dtype=torch.float32)

    for i in range(n_mels):
        left = int(bin_points[i].item())
        center = int(bin_points[i + 1].item())
        right = int(bin_points[i + 2].item())

        # Left slope: rising from 0 to 1
        for j in range(left, center):
            if center > left and j < num_bins:
                filterbank[i, j] = (j - left) / (center - left)

        # Right slope: falling from 1 to 0
        for j in range(center, right):
            if right > center and j < num_bins:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


class LogMelSpectrogram(nn.Module):
    """Extract Log-Mel spectrogram features from raw waveform tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels

        filterbank = create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.register_buffer("mel_filterbank", filterbank, persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram.

        Args:
            waveform: Tensor of shape (batch_size, num_samples) or (num_samples,).

        Returns:
            Log-mel spectrogram tensor of shape (batch_size, n_mels, num_frames).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device

        # Compute Short-Time Fourier Transform (STFT)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Magnitude spectrogram: (batch_size, 1 + n_fft // 2, num_frames)
        magnitude = torch.abs(stft)

        # Project onto Mel scale: (batch_size, n_mels, num_frames)
        mel_spectrogram = torch.matmul(self.mel_filterbank.to(device), magnitude)

        # Log compression with numerical stability clamp
        log_mel = torch.log(torch.clamp(mel_spectrogram, min=1e-5))

        return log_mel


def extract_log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> torch.Tensor:
    """Convenience helper to extract log-mel spectrogram features from a waveform tensor."""
    extractor = LogMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return extractor(waveform).squeeze(0)

