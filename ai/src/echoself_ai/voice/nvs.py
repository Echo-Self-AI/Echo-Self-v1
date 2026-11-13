"""Neural Voice Synthesis (NVS) — Week 4 EchoSelf.

High-level API that chains:
  CharacterTokenizer → AcousticModel → GriffinLimVocoder → validated waveform

Usage::

    from echoself_ai.voice.nvs import NVSConfig, TextToSpeech, synthesize_to_file

    tts = TextToSpeech()
    waveform, sr = tts.synthesize("hello world")
    synthesize_to_file("hello world", "out.wav")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from echoself_ai.tokenizer import CharacterTokenizer
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.audio import normalize_waveform, write_wav
from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.vocoder import GriffinLimVocoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default vocabulary — covers all printable ASCII characters.
# The tokenizer is built from this string; any character not present will
# raise a ValueError at encode time (clean, predictable error boundary).
# ---------------------------------------------------------------------------
_DEFAULT_VOCAB: str = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)


# ---------------------------------------------------------------------------
# NVSConfig
# ---------------------------------------------------------------------------


@dataclass
class NVSConfig:
    """Hyperparameters and paths for the Neural Voice Synthesis pipeline."""

    # Acoustic model
    vocab_size: int = len(set(_DEFAULT_VOCAB))  # derived from default vocab
    n_mels: int = 80
    hidden_dim: int = 64
    num_layers: int = 3
    num_speakers: int = 4
    upsample_factor: int = 4
    dropout: float = 0.1

    # Vocoder
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    n_iters: int = 32  # Griffin-Lim iterations (more → better quality, slower)

    # Checkpoint — loaded if it exists
    checkpoint_path: Optional[str] = str(
        Path(__file__).parent.parent.parent.parent.parent
        / "checkpoints"
        / "acoustic_real"
        / "best_model.pt"
    )

    # Amplitude validation
    max_amplitude: float = 1.0
    target_peak: float = 0.95

    # Device
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.sample_rate < 8000:
            raise ValueError(f"sample_rate must be ≥8000 Hz, got {self.sample_rate}")
        if not (0.0 < self.target_peak <= 1.0):
            raise ValueError(f"target_peak must be in (0, 1], got {self.target_peak}")

    def to_acoustic_config(self) -> AcousticModelConfig:
        return AcousticModelConfig(
            vocab_size=self.vocab_size,
            n_mels=self.n_mels,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_speakers=self.num_speakers,
            upsample_factor=self.upsample_factor,
            dropout=self.dropout,
        )


# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------


class TextToSpeech:
    """End-to-end Neural Voice Synthesis pipeline.

    Converts raw text strings to audible waveforms via:
        text → CharacterTokenizer → AcousticModel → GriffinLimVocoder → waveform
    """

    def __init__(
        self,
        config: Optional[NVSConfig] = None,
        tokenizer: Optional[CharacterTokenizer] = None,
    ) -> None:
        self.config = config or NVSConfig()
        self.device = torch.device(self.config.device)

        # Tokenizer — build from default ASCII vocab unless a custom one is provided
        self._tokenizer = tokenizer or CharacterTokenizer(_DEFAULT_VOCAB)

        # Ensure config vocab_size matches the actual tokenizer vocab
        acoustic_cfg = self.config.to_acoustic_config()
        acoustic_cfg.vocab_size = self._tokenizer.vocab_size

        # Acoustic model — load from checkpoint or create fresh
        self._acoustic_model: AcousticModel = self._load_or_create_acoustic_model(acoustic_cfg)
        self._acoustic_model.to(self.device).eval()

        # Vocoder
        self._vocoder = GriffinLimVocoder(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            n_iters=self.config.n_iters,
        )
        self._vocoder.to(self.device).eval()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_create_acoustic_model(self, cfg: AcousticModelConfig) -> AcousticModel:
        """Return an AcousticModel, loading weights from checkpoint when available."""
        model = AcousticModel(cfg)

        ckpt_path = self.config.checkpoint_path
        if ckpt_path is None:
            logger.info("No checkpoint path configured — using freshly initialized AcousticModel.")
            return model

        p = Path(ckpt_path)
        if not p.is_file():
            logger.warning(
                "Checkpoint not found at %s — using freshly initialized AcousticModel.", p
            )
            return model

        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
            # Support both raw state_dict and wrapped dicts (e.g. {"model_state_dict": ...})
            if isinstance(state, dict):
                state_dict = (
                    state.get("model_state_dict")
                    or state.get("state_dict")
                    or state.get("model")
                    or state
                )
            else:
                state_dict = state

            # Attempt strict load; fall back to partial load on shape mismatch
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("Loaded AcousticModel weights from %s (strict).", p)
            except RuntimeError as exc:
                logger.warning(
                    "Strict checkpoint load failed (%s). Attempting partial load.", exc
                )
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(
                    "Partial load complete — missing keys: %d, unexpected keys: %d.",
                    len(missing),
                    len(unexpected),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load checkpoint from %s (%s) — using freshly initialized model.",
                p,
                exc,
            )

        return model

    def _tokenize(self, text: str) -> torch.Tensor:
        """Encode text → (1, seq_len) int64 tensor on device."""
        ids = self._tokenizer.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _validate_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Check for NaN/Inf, clip amplitude, and peak-normalize."""
        if not torch.isfinite(waveform).all():
            logger.warning("Waveform contains NaN/Inf values — replacing with zeros.")
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)

        # Hard clip to configured max amplitude
        waveform = torch.clamp(waveform, -self.config.max_amplitude, self.config.max_amplitude)

        # Peak-normalize to target_peak
        waveform = normalize_waveform(waveform, target_peak=self.config.target_peak)

        return waveform

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        user_id: Optional[str] = None,
        consent_manager: Optional[VoiceConsentManager] = None,
    ) -> tuple[torch.Tensor, int]:
        """Synthesize speech from a text string.

        Args:
            text: Input text. Must only contain characters in the tokenizer vocabulary.
            speaker_id: Optional speaker index (0-indexed, up to num_speakers-1).
            user_id: User identifier used by the consent gate.
            consent_manager: Optional VoiceConsentManager; enforces consent before synthesis.

        Returns:
            (waveform, sample_rate) where waveform is a 1D float32 CPU tensor.

        Raises:
            ValueError: If text contains characters outside the tokenizer vocabulary.
            PermissionError: If consent_manager is provided but consent is not granted.
        """
        if not text:
            # Return a silent 0-length tensor; callers may treat this as a no-op.
            return torch.zeros(0, dtype=torch.float32), self.config.sample_rate

        token_ids = self._tokenize(text)

        # Build speaker tensor if requested
        spk_tensor: Optional[torch.Tensor] = None
        if speaker_id is not None:
            spk_tensor = torch.tensor([speaker_id], dtype=torch.long, device=self.device)

        # Consent gate — required when synthesizing a specific speaker's voice
        raw_speaker_id: Optional[str] = str(speaker_id) if speaker_id is not None else None

        self._acoustic_model.eval()
        self._vocoder.eval()

        with torch.no_grad():
            predicted_mel, _ = self._acoustic_model(
                token_ids=token_ids,
                speaker_id=spk_tensor,
                user_id=user_id,
                raw_speaker_id=raw_speaker_id,
                consent_manager=consent_manager if (consent_manager and raw_speaker_id) else None,
            )
            # predicted_mel: (1, n_mels, num_frames)
            waveform_batch = self._vocoder(predicted_mel)
            # waveform_batch: (1, num_samples)

        # Squeeze to 1D
        waveform = waveform_batch.squeeze(0)  # (num_samples,)

        # Validate and normalize
        waveform = self._validate_waveform(waveform)

        # Always return on CPU as float32
        return waveform.cpu().float(), self.config.sample_rate

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        speaker_id: Optional[int] = None,
        user_id: Optional[str] = None,
        consent_manager: Optional[VoiceConsentManager] = None,
    ) -> Path:
        """Synthesize speech and write the result to a WAV file.

        Args:
            text: Input text string.
            output_path: Destination file path (created if parents missing).
            speaker_id: Optional speaker index.
            user_id: User identifier for consent gate.
            consent_manager: Optional VoiceConsentManager.

        Returns:
            Resolved Path to the written WAV file.
        """
        waveform, sr = self.synthesize(
            text,
            speaker_id=speaker_id,
            user_id=user_id,
            consent_manager=consent_manager,
        )
        out = Path(output_path).resolve()
        write_wav(out, waveform, sample_rate=sr)
        logger.info("Wrote synthesized speech to %s (%d samples @ %d Hz).", out, len(waveform), sr)
        return out

    def get_synthesis_latency(self, text: str, speaker_id: Optional[int] = None) -> float:
        """Measure wall-clock time (seconds) to synthesize the given text.

        Returns:
            Elapsed seconds as a positive float.
        """
        t0 = time.perf_counter()
        self.synthesize(text, speaker_id=speaker_id)
        return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def synthesize_to_file(
    text: str,
    output_path: Union[str, Path],
    config: Optional[NVSConfig] = None,
    speaker_id: Optional[int] = None,
    user_id: Optional[str] = None,
    consent_manager: Optional[VoiceConsentManager] = None,
) -> Path:
    """Convenience function: create a TextToSpeech instance and write output to a file.

    Args:
        text: Input text string.
        output_path: Destination file path.
        config: Optional NVSConfig; defaults are used if not provided.
        speaker_id: Optional speaker index.
        user_id: User identifier for consent gate.
        consent_manager: Optional VoiceConsentManager.

    Returns:
        Resolved Path to the written WAV file.
    """
    tts = TextToSpeech(config=config)
    return tts.synthesize_to_file(
        text,
        output_path,
        speaker_id=speaker_id,
        user_id=user_id,
        consent_manager=consent_manager,
    )


def get_synthesis_latency(
    text: str,
    config: Optional[NVSConfig] = None,
    speaker_id: Optional[int] = None,
) -> float:
    """Module-level latency measurement.  Creates a TextToSpeech and times one synthesis call."""
    tts = TextToSpeech(config=config)
    return tts.get_synthesis_latency(text, speaker_id=speaker_id)
