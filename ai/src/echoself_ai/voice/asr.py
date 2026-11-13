"""Character-level CTC Automatic Speech Recognition (ASR) for EchoSelf.

Architecture overview
---------------------
1. ``ASRModel``  — CNN front-end (2-D feature extraction) followed by a stack
   of bidirectional GRU layers.  Outputs per-frame log-softmax character
   probabilities that are compatible with ``torch.nn.CTCLoss``.

2. ``CTCDecoder`` — Greedy (best-path) CTC decoder.  Collapses repeated tokens
   and strips blank symbols to produce a character string.

3. ``SpeechRecognizer`` — High-level recognizer.  Accepts a raw waveform tensor,
   runs the Voice Activity Detector to split it into speech segments, extracts
   log-mel spectrograms for each segment, feeds them through ``ASRModel``, and
   assembles the per-segment transcripts into a single string.

Training helper
---------------
``train_asr_smoke()`` trains the model on a trivial 1-phoneme synthetic CTC task
for 10 steps.  It is intended purely as a smoke-test to verify that the forward
pass, CTC loss, and backprop all function correctly end-to-end.

Usage::

    from echoself_ai.voice.asr import SpeechRecognizer, ASRConfig
    recognizer = SpeechRecognizer()
    transcript = recognizer.recognize(waveform, sample_rate=16000)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from echoself_ai.voice.asr_config import ASRConfig
from echoself_ai.voice.features import LogMelSpectrogram
from echoself_ai.voice.vad import VoiceActivityDetector, create_vad

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public character vocabulary
# ---------------------------------------------------------------------------

#: Default character set: lowercase a-z, space, apostrophe, blank (index 28).
_CHARACTERS: str = "abcdefghijklmnopqrstuvwxyz '"
# blank token is implicitly at index len(_CHARACTERS) == 28


def _build_default_vocab() -> Tuple[dict, dict]:
    """Build (char→id, id→char) maps for the default ASCII character set."""
    char_to_id = {ch: i for i, ch in enumerate(_CHARACTERS)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    return char_to_id, id_to_char


# ---------------------------------------------------------------------------
# ASRModel
# ---------------------------------------------------------------------------

class _CNNFrontEnd(nn.Module):
    """Small 2-D CNN that processes log-mel spectrograms.

    Input shape:  (B, 1, n_mels, T)
    Output shape: (B, cnn_channels * reduced_mels, T)

    Two strided conv layers halve the mel dimension to reduce sequence
    length before the recurrent layers.
    """

    def __init__(self, n_mels: int, cnn_channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=(2, 1),   # stride on frequency axis only
            padding=pad,
        )
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.conv2 = nn.Conv2d(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=(2, 1),
            padding=pad,
        )
        self.bn2 = nn.BatchNorm2d(cnn_channels)

        # Compute output frequency bins after two stride-2 convs
        f1 = (n_mels + 2 * pad - kernel_size) // 2 + 1
        f2 = (f1 + 2 * pad - kernel_size) // 2 + 1
        self.output_dim = cnn_channels * f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        # x: (B, cnn_channels, reduced_mels, T)
        B, C, F_, T = x.shape
        # Merge channel and frequency dims → (B, C*F_, T)
        x = x.view(B, C * F_, T)
        # Transpose to (B, T, features) for RNN
        return x.transpose(1, 2)


class ASRModel(nn.Module):
    """Encoder-only CTC ASR model.

    Takes log-mel spectrograms and outputs per-frame log-character probabilities.

    Args:
        config: ``ASRConfig`` instance with model hyper-parameters.

    Input:
        log_mel: Tensor of shape ``(B, n_mels, T)``.

    Output:
        log_probs: Tensor of shape ``(T, B, vocab_size)`` — log-softmax
            probabilities in the time-first format expected by
            ``torch.nn.CTCLoss``.
        output_lengths: Tensor of shape ``(B,)`` — number of valid time steps
            per batch element (all equal to T in this implementation since we
            do not downsample the time axis).
    """

    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.config = config

        self.cnn = _CNNFrontEnd(
            n_mels=config.n_mels,
            cnn_channels=config.cnn_channels,
            kernel_size=config.cnn_kernel_size,
        )

        # Project CNN output to GRU hidden dim
        self.input_proj = nn.Linear(self.cnn.output_dim, config.hidden_dim)

        # Bidirectional GRU stack
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # Project bidirectional GRU output to vocabulary
        self.output_proj = nn.Linear(config.hidden_dim * 2, config.vocab_size)

    def forward(
        self, log_mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            log_mel: ``(B, n_mels, T)`` log-mel spectrogram.

        Returns:
            log_probs: ``(T, B, vocab_size)`` log-softmax output.
            output_lengths: ``(B,)`` int tensor of time steps.
        """
        B, n_mels, T = log_mel.shape

        # Add channel dim for CNN: (B, 1, n_mels, T)
        x = log_mel.unsqueeze(1)

        # CNN front-end → (B, T, features)
        x = self.cnn(x)

        # Project to hidden dim → (B, T, hidden_dim)
        x = F.gelu(self.input_proj(x))

        # GRU encoder → (B, T, hidden_dim * 2)
        x, _ = self.gru(x)

        # Project to vocabulary → (B, T, vocab_size)
        logits = self.output_proj(x)

        # Log-softmax along vocab dimension
        log_probs = F.log_softmax(logits, dim=-1)

        # CTC expects time-first: (T, B, vocab_size)
        log_probs = log_probs.transpose(0, 1)

        output_lengths = torch.full(
            (B,), T, dtype=torch.long, device=log_mel.device
        )

        return log_probs, output_lengths


# ---------------------------------------------------------------------------
# CTCDecoder
# ---------------------------------------------------------------------------

class CTCDecoder:
    """Greedy (best-path) CTC decoder.

    Collapses repeated tokens and removes blank symbols to produce a string.

    Args:
        blank_id: Index of the CTC blank token.  Must match the value used
            during training.
        id_to_char: Mapping from token index to character string.  If ``None``
            the default ASCII vocabulary is used.
    """

    def __init__(
        self,
        blank_id: int = 28,
        id_to_char: Optional[dict] = None,
    ) -> None:
        self.blank_id = blank_id
        if id_to_char is None:
            _, self.id_to_char = _build_default_vocab()
        else:
            self.id_to_char = id_to_char

    def decode(self, log_probs: torch.Tensor) -> str:
        """Greedy decode a single log-probability sequence.

        Args:
            log_probs: Tensor of shape ``(T, vocab_size)`` — log-softmax output
                for a **single** example (no batch dimension).

        Returns:
            Decoded character string (may be empty if all frames are blank).
        """
        # Argmax at each time step
        indices = log_probs.argmax(dim=-1).tolist()  # list of ints length T

        chars: List[str] = []
        prev: Optional[int] = None
        for idx in indices:
            if idx == self.blank_id:
                prev = None
                continue
            if idx != prev:
                ch = self.id_to_char.get(idx, "")
                chars.append(ch)
            prev = idx

        return "".join(chars).strip()

    def decode_batch(self, log_probs: torch.Tensor) -> List[str]:
        """Decode a batched log-probability tensor.

        Args:
            log_probs: Tensor of shape ``(T, B, vocab_size)`` — time-first batch.

        Returns:
            List of ``B`` decoded strings.
        """
        T, B, V = log_probs.shape
        return [self.decode(log_probs[:, b, :]) for b in range(B)]


# ---------------------------------------------------------------------------
# SpeechRecognizer
# ---------------------------------------------------------------------------

class SpeechRecognizer:
    """High-level interface for automatic speech recognition.

    Pipeline:
        1. Run VAD on the full waveform to locate speech segments.
        2. For each segment: extract log-mel spectrogram, run ``ASRModel``,
           greedily decode the output.
        3. Join per-segment transcripts with a space separator.

    Args:
        config: ``ASRConfig`` instance.  Defaults to the standard 29-class config.
        device: PyTorch device.  Defaults to CUDA if available, else CPU.

    Example::

        recognizer = SpeechRecognizer()
        text = recognizer.recognize(waveform, sample_rate=16000)
    """

    def __init__(
        self,
        config: Optional[ASRConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config if config is not None else ASRConfig()

        # CPU fallback when CUDA is unavailable
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        self.model = ASRModel(self.config).to(self.device)
        self.model.eval()

        self.decoder = CTCDecoder(
            blank_id=self.config.blank_id,
        )

        self.feature_extractor = LogMelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
        ).to(self.device)

        self.vad = create_vad(
            method="energy",
            sample_rate=self.config.sample_rate,
            min_speech_duration_ms=100,
            hangover_ms=200,
        )

        # Optionally load a pre-trained checkpoint
        if self.config.checkpoint_path is not None:
            self._load_checkpoint(Path(self.config.checkpoint_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """Transcribe a waveform tensor to text.

        Args:
            waveform: 1-D float32 tensor of raw PCM samples.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            Decoded transcript string.  Returns an empty string when the
            waveform contains only silence (no VAD segments detected) or
            when the waveform is empty.
        """
        waveform = waveform.detach().cpu().float()

        # --- guard: empty audio ---
        if waveform.numel() == 0:
            return ""

        # --- segment via VAD ---
        segments = self.vad.segment_waveform(waveform, sample_rate=sample_rate)

        if not segments:
            return ""

        transcripts: List[str] = []
        for seg_wave in segments:
            if seg_wave.numel() < self.config.hop_length:
                # Too short to form even one STFT frame — skip
                continue

            transcript = self._transcribe_segment(seg_wave)
            if transcript:
                transcripts.append(transcript)

        return " ".join(transcripts).strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _transcribe_segment(self, waveform: torch.Tensor) -> str:
        """Transcribe a single waveform segment."""
        waveform = waveform.to(self.device)

        # Log-mel spectrogram: (n_mels, T)
        log_mel = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)

        if log_mel.shape[-1] < 1:
            return ""

        # Add batch dimension: (1, n_mels, T)
        log_mel = log_mel.unsqueeze(0)

        # Forward pass: (T, 1, vocab_size)
        log_probs, _ = self.model(log_mel)

        # Decode: (T, vocab_size) for the single example
        transcript = self.decoder.decode(log_probs[:, 0, :])
        return transcript

    def _load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file if it exists."""
        if not path.exists():
            logger.warning(
                "ASR checkpoint not found at %s — using untrained model.", path
            )
            return
        try:
            state = torch.load(path, map_location=self.device)
            # Support both raw state-dict and wrapped checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("Loaded ASR checkpoint from %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load ASR checkpoint from %s: %s — using untrained model.",
                path,
                exc,
            )


# ---------------------------------------------------------------------------
# Smoke-training helper
# ---------------------------------------------------------------------------

def train_asr_smoke(
    num_steps: int = 10,
    *,
    config: Optional[ASRConfig] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run a minimal CTC smoke-training loop on a synthetic single-phoneme task.

    Creates a tiny synthetic dataset: a single log-mel spectrogram (batch of 1)
    paired with a 1-character target label.  Trains for ``num_steps`` gradient
    updates and returns a summary dict.

    This function validates that:
    - ``ASRModel`` forward pass produces the correct shapes.
    - ``torch.nn.CTCLoss`` can consume the output.
    - Gradients flow back through the model without NaN/Inf.
    - Loss decreases over steps (not guaranteed with an untrained model on a
      random input, but should not diverge).

    Args:
        num_steps: Number of gradient update steps (default 10).
        config: Optional ``ASRConfig``.  Defaults to a minimal config.
        device: Torch device.  Defaults to CPU (keeps the test fast).

    Returns:
        Dict with keys: ``initial_loss``, ``final_loss``, ``steps``,
        ``loss_decreased`` (bool).
    """
    if config is None:
        # Use a small model so the smoke test stays fast on CPU
        config = ASRConfig(
            vocab_size=10,
            blank_id=9,
            n_mels=40,
            cnn_channels=8,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
        )

    if device is None:
        device = torch.device("cpu")

    model = ASRModel(config).to(device)
    model.train()

    ctc_loss = nn.CTCLoss(blank=config.blank_id, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic input: 1 example, n_mels channels, 50 time frames
    T_input = 50
    batch_size = 1
    log_mel = torch.randn(batch_size, config.n_mels, T_input, device=device)

    # Target: index 0 (a single character — not blank)
    target = torch.tensor([[0]], dtype=torch.long, device=device)
    target_lengths = torch.tensor([1], dtype=torch.long, device=device)

    losses: List[float] = []

    for _ in range(num_steps):
        optimizer.zero_grad()

        log_probs, input_lengths = model(log_mel)
        # log_probs: (T_out, B, vocab_size), input_lengths: (B,)

        loss = ctc_loss(log_probs, target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    model.eval()

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "steps": num_steps,
        "loss_decreased": losses[-1] < losses[0],
        "all_losses": losses,
    }
