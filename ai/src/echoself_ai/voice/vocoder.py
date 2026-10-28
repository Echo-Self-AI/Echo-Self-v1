import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import write_wav
from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.features import create_mel_filterbank


class GriffinLimVocoder(nn.Module):
    """Zero-dependency Griffin-Lim phase reconstruction algorithm.

    Inverts log-mel spectrogram representations back to audible time-domain
    waveforms using iterative STFT / iSTFT phase estimation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_mels: int = 80,
        n_iters: int = 32,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iters = n_iters

        # Forward mel filterbank
        fb = create_mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("mel_filterbank", fb, persistent=False)

        # Moore-Penrose pseudo-inverse to map Mel energies back to linear FFT magnitude bins
        inv_fb = torch.pinverse(fb)
        self.register_buffer("inv_mel_filterbank", inv_fb, persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def mel_to_linear(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert log-mel spectrogram to linear magnitude spectrogram."""
        # log_mel shape: (B, n_mels, T)
        mel_linear = torch.exp(log_mel)
        device = log_mel.device
        # (B, num_bins, T) = (257, 80) @ (B, 80, T)
        linear_mag = torch.matmul(self.inv_mel_filterbank.to(device), mel_linear)
        return torch.clamp(linear_mag, min=1e-5)


    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Reconstruct time-domain waveform from log-mel spectrogram.

        Args:
            log_mel: Tensor of shape (batch_size, n_mels, num_frames) or (n_mels, num_frames).

        Returns:
            Reconstructed waveform tensor of shape (batch_size, num_samples).
        """
        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)

        device = log_mel.device
        linear_mag = self.mel_to_linear(log_mel)  # (B, 1 + n_fft//2, T)

        # Initial random phase estimation
        angles = torch.rand_like(linear_mag) * 2.0 * math.pi - math.pi
        complex_spec = torch.polar(linear_mag, angles)

        window = self.window.to(device)

        # Iterative phase approximation
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            onesided=True,
        )

        for _ in range(self.n_iters):
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                onesided=True,
                return_complex=True,
            )
            # Retain original target linear magnitude, update estimated phase angle
            angles = torch.angle(stft)
            min_frames = min(linear_mag.size(-1), angles.size(-1))
            complex_spec = torch.polar(
                linear_mag[:, :, :min_frames],
                angles[:, :, :min_frames],
            )
            waveform = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                onesided=True,
            )

        return waveform


class VoiceSynthesizer:
    """Unified Voice Synthesis pipeline chaining AcousticModel and GriffinLimVocoder."""

    def __init__(
        self,
        acoustic_model: AcousticModel,
        vocoder: Optional[GriffinLimVocoder] = None,
        consent_manager: Optional[VoiceConsentManager] = None,
    ) -> None:
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder or GriffinLimVocoder(
            sample_rate=16000,
            n_fft=512,
            hop_length=128,
            win_length=512,
            n_mels=acoustic_model.config.n_mels,
            n_iters=24,
        )
        self.consent_manager = consent_manager

    def synthesize(
        self,
        token_ids: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        user_id: Optional[str] = None,
        raw_speaker_id: Optional[str] = None,
        output_wav_path: Optional[str | Path] = None,
    ) -> torch.Tensor:
        """Synthesize audible speech waveform from input token sequence."""
        self.acoustic_model.eval()
        self.vocoder.eval()

        with torch.no_grad():
            predicted_mel, _ = self.acoustic_model(
                token_ids=token_ids,
                speaker_id=speaker_id,
                user_id=user_id,
                raw_speaker_id=raw_speaker_id,
                consent_manager=self.consent_manager if raw_speaker_id else None,
            )
            waveform = self.vocoder(predicted_mel)


        if output_wav_path is not None:
            write_wav(output_wav_path, waveform[0], self.vocoder.sample_rate)

        return waveform
