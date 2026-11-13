"""
Learned audio watermarking using a lightweight encoder-decoder architecture.

Architecture overview
---------------------
WatermarkEncoder
    Takes an audio segment and a binary message, embeds the message as a
    small perceptual perturbation using convolutional layers.

WatermarkDecoder
    Takes a (possibly watermarked) audio segment, extracts bit logits via
    convolutional feature extraction followed by global average pooling.

LearnedWatermarker
    High-level class that owns the encoder/decoder, handles training, and
    provides the same embed/extract interface as SpreadSpectrumWatermarker.

Training
--------
Trains on random synthetic audio (uniform noise) for fast convergence.
Loss is BCE between the decoded bit logits and the original bits.
200 steps on a modern CPU is sufficient to verify the architecture learns
to route message bits through audio (accuracy > 50 % random baseline).

Only PyTorch is required — no scipy, librosa, or other external deps.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LearnedWatermarkConfig:
    """Configuration for the learned audio watermarking model.

    Attributes:
        n_bits: Number of message bits to embed / extract.
        hidden_dim: Number of hidden channels in convolutional layers.
        message_dim: Alias kept for API symmetry; equals n_bits in practice.
        sample_rate: Expected audio sample rate in Hz.
        segment_length: Number of samples per segment (default 1 s at 16 kHz).
    """

    n_bits: int = 16
    hidden_dim: int = 64
    message_dim: int = 16
    sample_rate: int = 16000
    segment_length: int = 16000


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class WatermarkEncoder(nn.Module):
    """Embed a binary message into an audio segment.

    Args:
        config: :class:`LearnedWatermarkConfig` instance.

    Input:
        audio_segment : ``(B, segment_length)`` float32
        message_bits  : ``(B, n_bits)``          float32  (0.0 or 1.0)

    Output:
        watermarked audio : ``(B, segment_length)`` float32
    """

    def __init__(self, config: LearnedWatermarkConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Audio feature extraction: (B, 1, T) → (B, H, T)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.Conv1d(32, H, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        # Message embedding: (B, n_bits) → (B, H)
        self.message_linear = nn.Linear(config.n_bits, H)

        # Output reconstruction: (B, H, T) → (B, 1, T)
        self.output_conv = nn.Sequential(
            nn.Conv1d(H, 32, kernel_size=7, padding=3),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(
        self,
        audio_segment: torch.Tensor,
        message_bits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            audio_segment: ``(B, segment_length)``
            message_bits:  ``(B, n_bits)``

        Returns:
            Watermarked audio of shape ``(B, segment_length)``.
        """
        B, T = audio_segment.shape

        # Audio path: (B, T) → (B, 1, T) → (B, H, T)
        x = audio_segment.unsqueeze(1)          # (B, 1, T)
        audio_features = self.audio_conv(x)     # (B, H, T)

        # Message path: (B, n_bits) → (B, H) → (B, H, 1) → (B, H, T)
        msg_emb = self.message_linear(message_bits)      # (B, H)
        msg_emb = msg_emb.unsqueeze(2).expand(-1, -1, T) # (B, H, T)

        # Combine by addition, then decode back to audio delta
        combined = audio_features + msg_emb     # (B, H, T)
        delta = self.output_conv(combined)      # (B, 1, T)
        delta = delta.squeeze(1)                # (B, T)

        # Small perturbation so the watermark is perceptually minimal
        watermarked = audio_segment + torch.tanh(delta) * 0.01
        return watermarked


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class WatermarkDecoder(nn.Module):
    """Extract bit logits from an audio segment.

    Args:
        config: :class:`LearnedWatermarkConfig` instance.

    Input:
        audio_segment : ``(B, segment_length)`` float32

    Output:
        logits : ``(B, n_bits)`` float32  — raw logits for BCE loss
    """

    def __init__(self, config: LearnedWatermarkConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # Feature extraction: (B, 1, T) → (B, H, T)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.Conv1d(32, H, kernel_size=7, padding=3),
        )

        # Global average pool collapses the time dimension: (B, H, T) → (B, H)
        # then project to bit logits
        self.classifier = nn.Linear(H, config.n_bits)

    def forward(self, audio_segment: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio_segment: ``(B, segment_length)``

        Returns:
            Logits of shape ``(B, n_bits)``.
        """
        x = audio_segment.unsqueeze(1)  # (B, 1, T)
        features = self.conv(x)         # (B, H, T)
        pooled = features.mean(dim=2)   # (B, H)  — global average pool
        logits = self.classifier(pooled) # (B, n_bits)
        return logits


# ---------------------------------------------------------------------------
# High-level watermarker
# ---------------------------------------------------------------------------

class LearnedWatermarker:
    """High-level learned audio watermarker.

    Owns the :class:`WatermarkEncoder` and :class:`WatermarkDecoder`, handles
    model training and inference.

    Args:
        config: :class:`LearnedWatermarkConfig` instance.  Defaults to the
                standard 16-bit / 64-dim config.
        checkpoint_dir: Directory for saving trained model checkpoints.
                        Defaults to ``ai/checkpoints/learned_watermark/``
                        relative to the repository root (auto-detected via
                        this file's location, with a fallback to cwd).
    """

    def __init__(
        self,
        config: Optional[LearnedWatermarkConfig] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.config = config or LearnedWatermarkConfig()
        self.encoder = WatermarkEncoder(self.config)
        self.decoder = WatermarkDecoder(self.config)
        self._trained = False

        # Resolve checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            # Navigate up from this file to find the repo ai/ directory
            # learned.py sits at ai/src/echoself_ai/watermark/learned.py
            this_file = Path(__file__).resolve()
            # Go up 4 levels: watermark → echoself_ai → src → ai
            ai_dir = this_file.parents[3]
            self.checkpoint_dir = ai_dir / "checkpoints" / "learned_watermark"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        n_steps: int = 200,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
    ) -> Dict:
        """Train encoder and decoder on synthetic random audio.

        Generates random audio and random binary messages, embeds, then
        decodes.  Optimises BCE loss so the decoder can recover the bits.

        Args:
            n_steps: Number of gradient-update steps.
            batch_size: Batch size (number of segments per step).
            learning_rate: Adam learning rate.

        Returns:
            Dictionary with keys:
                - ``encoder_path`` (str): Path to saved encoder checkpoint.
                - ``decoder_path`` (str): Path to saved decoder checkpoint.
                - ``final_loss`` (float): Loss at the last training step.
                - ``train_losses`` (list[float]): Loss at every step.
        """
        self.encoder.train()
        self.decoder.train()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)
        bce_loss = nn.BCEWithLogitsLoss()

        train_losses: List[float] = []
        T = self.config.segment_length
        n_bits = self.config.n_bits

        for step in range(n_steps):
            # Random synthetic audio in [-1, 1]
            audio = torch.rand(batch_size, T) * 2.0 - 1.0
            # Random binary messages (float for BCE)
            bits = torch.randint(0, 2, (batch_size, n_bits)).float()

            # Forward
            watermarked = self.encoder(audio, bits)
            logits = self.decoder(watermarked)

            loss = bce_loss(logits, bits)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        self.encoder.eval()
        self.decoder.eval()
        self._trained = True

        # Save checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = str(self.checkpoint_dir / "encoder.pt")
        decoder_path = str(self.checkpoint_dir / "decoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

        return {
            "encoder_path": encoder_path,
            "decoder_path": decoder_path,
            "final_loss": train_losses[-1],
            "train_losses": train_losses,
        }

    def load_model(self, encoder_path: str, decoder_path: str) -> None:
        """Load pre-trained encoder/decoder weights.

        Args:
            encoder_path: Path to encoder state-dict ``.pt`` file.
            decoder_path: Path to decoder state-dict ``.pt`` file.
        """
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        self.encoder.eval()
        self.decoder.eval()
        self._trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def embed(
        self,
        waveform: torch.Tensor,
        message_bits: torch.Tensor,
    ) -> torch.Tensor:
        """Embed *message_bits* into *waveform* using the trained encoder.

        The waveform is processed in :attr:`~LearnedWatermarkConfig.segment_length`
        chunks.  Any trailing samples shorter than a full segment are passed
        through unchanged (the perturbation is too small to matter at the
        boundary).

        Args:
            waveform: 1-D float32 audio tensor of arbitrary length.
            message_bits: 1-D float32 tensor of shape ``(n_bits,)`` with
                          values in {0.0, 1.0}.

        Returns:
            Watermarked waveform of the same shape as *waveform*.
        """
        self.encoder.eval()
        T = self.config.segment_length
        waveform = waveform.float()
        total = waveform.numel()
        output = waveform.clone()

        # Expand message bits to batch dimension
        bits = message_bits.float().unsqueeze(0)  # (1, n_bits)

        with torch.no_grad():
            start = 0
            while start + T <= total:
                seg = waveform[start : start + T].unsqueeze(0)  # (1, T)
                watermarked_seg = self.encoder(seg, bits)        # (1, T)
                output[start : start + T] = watermarked_seg.squeeze(0)
                start += T

        return output

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract bit probabilities from *waveform*.

        Processes the first full segment; if the waveform is shorter than
        ``segment_length``, it is zero-padded.

        Args:
            waveform: 1-D float32 audio tensor.

        Returns:
            Bit probability tensor of shape ``(n_bits,)`` with values in
            [0, 1], obtained by applying sigmoid to the decoder logits.
        """
        self.decoder.eval()
        T = self.config.segment_length
        waveform = waveform.float()
        total = waveform.numel()

        if total >= T:
            seg = waveform[:T].unsqueeze(0)  # (1, T)
        else:
            # Zero-pad to segment length
            seg = torch.zeros(1, T)
            seg[0, :total] = waveform

        with torch.no_grad():
            logits = self.decoder(seg)          # (1, n_bits)
            probs = torch.sigmoid(logits)       # (1, n_bits)

        return probs.squeeze(0)  # (n_bits,)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def compute_bit_accuracy(
        self,
        original_bits: torch.Tensor,
        extracted_bits: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Fraction of bits that were correctly recovered.

        Args:
            original_bits: Ground-truth bits tensor; values in {0, 1} (or
                           {0.0, 1.0}).
            extracted_bits: Soft bit probabilities (output of
                            :meth:`extract`) or hard bits; shape must match
                            *original_bits*.
            threshold: Decision boundary applied to *extracted_bits* to
                       obtain hard binary decisions.

        Returns:
            Accuracy in [0.0, 1.0].
        """
        orig = original_bits.float()
        extr = extracted_bits.float()

        # Hard decisions
        hard = (extr >= threshold).float()
        orig_hard = (orig >= threshold).float()

        correct = (hard == orig_hard).float().sum().item()
        return correct / max(orig_hard.numel(), 1)

    def compare_with_classical(
        self,
        waveform: torch.Tensor,
        watermark_id: str,
    ) -> Dict:
        """Compare learned and classical watermarking on the same waveform.

        Embeds and extracts a watermark using both methods, returning
        accuracy / detection metrics side-by-side.

        Args:
            waveform: 1-D float32 audio tensor.
            watermark_id: String identifier used by the classical method.

        Returns:
            Dictionary with two top-level keys:

            * ``'classical'``: dict with keys ``detected``, ``correlation``,
              ``ber``.
            * ``'learned'``: dict with keys ``bit_accuracy``,
              ``mean_confidence``, ``n_bits``.
        """
        # --- Classical ---
        # Import here to avoid a circular dependency at module level.
        from .classical import SpreadSpectrumWatermarker, n_bit_watermark_id_to_bits
        from .config import ClassicalWatermarkConfig

        classical_cfg = ClassicalWatermarkConfig(
            watermark_id=watermark_id,
            n_bits=self.config.n_bits,
        )
        classical_wm = SpreadSpectrumWatermarker(classical_cfg)
        c_watermarked = classical_wm.embed(waveform, watermark_id)
        detected, correlation, ber = classical_wm.extract(c_watermarked, watermark_id)

        classical_result = {
            "detected": detected,
            "correlation": correlation,
            "ber": ber,
        }

        # --- Learned ---
        # Derive message bits from watermark_id for a fair comparison
        id_bits_list = n_bit_watermark_id_to_bits(watermark_id, self.config.n_bits)
        message_bits = torch.tensor(id_bits_list, dtype=torch.float32)

        l_watermarked = self.embed(waveform, message_bits)
        extracted_probs = self.extract(l_watermarked)
        bit_acc = self.compute_bit_accuracy(message_bits, extracted_probs)

        learned_result = {
            "bit_accuracy": bit_acc,
            "mean_confidence": float(extracted_probs.mean().item()),
            "n_bits": self.config.n_bits,
        }

        return {
            "classical": classical_result,
            "learned": learned_result,
        }
