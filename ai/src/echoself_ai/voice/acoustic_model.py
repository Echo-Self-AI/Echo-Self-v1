from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from echoself_ai.voice.consent import VoiceConsentManager


@dataclass
class AcousticModelConfig:
    """Hyperparameters for neural acoustic feature prediction."""

    vocab_size: int = 50
    n_mels: int = 80
    hidden_dim: int = 64
    num_layers: int = 3
    num_speakers: int = 4
    upsample_factor: int = 4  # Each text token expands to upsample_factor mel frames
    dropout: float = 0.1


class ConvBlock(nn.Module):
    """1D Convolution block with LayerNorm, activation, and residual connection."""

    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, hidden_dim, T)
        residual = x
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + residual


class AcousticModel(nn.Module):
    """Neural acoustic model predicting 80-channel mel-spectrogram frames from text tokens.

    Supports optional speaker ID conditioning for authorized personalized voice synthesis.
    """

    def __init__(self, config: AcousticModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.speaker_embedding = nn.Embedding(config.num_speakers, config.hidden_dim)

        self.encoder_layers = nn.ModuleList(
            [ConvBlock(config.hidden_dim, dropout=config.dropout) for _ in range(config.num_layers)]
        )

        # Upsample projection from token resolution to audio frame resolution
        self.upsample = nn.ConvTranspose1d(
            config.hidden_dim,
            config.hidden_dim,
            kernel_size=config.upsample_factor,
            stride=config.upsample_factor,
        )

        # Projection from hidden features to 80-channel log-mel spectrogram
        self.mel_proj = nn.Conv1d(config.hidden_dim, config.n_mels, kernel_size=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        speaker_id: Optional[torch.Tensor] = None,
        target_mel: Optional[torch.Tensor] = None,
        user_id: Optional[str] = None,
        raw_speaker_id: Optional[str] = None,
        consent_manager: Optional[VoiceConsentManager] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict mel-spectrogram frames from text tokens.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len).
            speaker_id: Optional speaker index tensor of shape (batch_size,).
            target_mel: Optional ground truth mel of shape (batch_size, n_mels, frames).
            user_id: User identifier for consent verification.
            raw_speaker_id: Speaker identifier for consent verification.
            consent_manager: Optional VoiceConsentManager for security enforcement.

        Returns:
            predicted_mel: Tensor of shape (batch_size, n_mels, frames).
            loss: Optional L1 reconstruction loss against target_mel.
        """
        # Strict consent gate enforcement
        if consent_manager is not None:
            if user_id is None or raw_speaker_id is None:
                raise PermissionError("Consent manager requires user_id and raw_speaker_id")
            consent_manager.enforce_gate(user_id, raw_speaker_id, scope="synthesis")

        # (batch_size, seq_len, hidden_dim)
        x = self.token_embedding(token_ids)

        if speaker_id is not None:
            # Condition on speaker embedding
            spk_emb = self.speaker_embedding(speaker_id).unsqueeze(1)
            x = x + spk_emb

        # (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)

        for layer in self.encoder_layers:
            x = layer(x)

        # Upsample along time dimension: (batch_size, hidden_dim, seq_len * upsample_factor)
        frames = self.upsample(x)

        # Project to mel channels: (batch_size, n_mels, num_frames)
        predicted_mel = self.mel_proj(frames)

        loss = None
        if target_mel is not None:
            # Match lengths if slight discrepancy
            min_len = min(predicted_mel.size(-1), target_mel.size(-1))
            loss = F.l1_loss(
                predicted_mel[:, :, :min_len],
                target_mel[:, :, :min_len],
            )

        return predicted_mel, loss
