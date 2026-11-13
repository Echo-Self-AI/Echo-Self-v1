"""Voice Activity Detection (VAD) for EchoSelf.

Provides two backends:
  - EnergyVAD: fast, threshold-based log-energy detector (pure PyTorch, no model).
  - NeuralVAD: lightweight LSTM-based classifier inspired by Silero VAD, trained
    entirely on synthetic data at construction time — no external downloads required.

Both backends are exposed through the unified ``VoiceActivityDetector`` interface.
Use ``create_vad()`` as the recommended factory entry point.

Example::

    from echoself_ai.voice.vad import create_vad
    from echoself_ai.voice.audio import generate_synthetic_tone

    vad = create_vad(method='energy')
    tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=1.0, sample_rate=16000)
    segments = vad.detect(tone, sample_rate=16000)
    print(segments)   # [(0.0, 1.0)]
"""
from __future__ import annotations

import math
from typing import Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from echoself_ai.voice.vad_config import VADConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frame_waveform(waveform: torch.Tensor, frame_samples: int) -> torch.Tensor:
    """Split a 1-D waveform into overlapping frames of ``frame_samples`` length.

    Returns:
        Tensor of shape (num_frames, frame_samples). The last incomplete frame
        is zero-padded so that all frames have equal length.
    """
    if waveform.numel() == 0:
        return waveform.new_zeros((0, frame_samples))

    # Pad to a multiple of frame_samples
    remainder = waveform.numel() % frame_samples
    if remainder != 0:
        pad_len = frame_samples - remainder
        waveform = F.pad(waveform, (0, pad_len))

    num_frames = waveform.numel() // frame_samples
    return waveform.view(num_frames, frame_samples)


def _frame_log_energy(frames: torch.Tensor) -> torch.Tensor:
    """Compute per-frame log-energy.

    Args:
        frames: (num_frames, frame_samples)

    Returns:
        (num_frames,) log energy values.
    """
    energy = (frames ** 2).mean(dim=1)
    return torch.log(energy + 1e-10)


def _apply_hangover_and_min_duration(
    frame_flags: torch.Tensor,
    hangover_frames: int,
    min_speech_frames: int,
) -> torch.Tensor:
    """Post-process binary frame flags with hangover and minimum-duration filtering.

    Args:
        frame_flags: (num_frames,) boolean/float tensor, 1 = speech.
        hangover_frames: Number of trailing frames to extend each speech region.
        min_speech_frames: Minimum run length to keep.

    Returns:
        Processed (num_frames,) boolean tensor.
    """
    flags = frame_flags.bool()
    n = len(flags)
    if n == 0:
        return flags

    # Apply hangover: extend each True run by hangover_frames.
    # We use the *original* flags to find run ends so that hangover does not
    # cascade (a frame set True by hangover does not start a new hangover).
    if hangover_frames > 0:
        original = flags.tolist()
        flags_list = list(original)  # mutable copy
        for i in range(n):
            if original[i]:
                # Extend forward only from the last frame of each run
                # (or from every speech frame — both are equivalent; we
                # just must not use the mutated list to decide the source)
                end = min(n, i + 1 + hangover_frames)
                for j in range(i + 1, end):
                    flags_list[j] = True
        flags = torch.tensor(flags_list, dtype=torch.bool)

    # Remove runs shorter than min_speech_frames
    if min_speech_frames > 1:
        flags_list = flags.tolist()
        i = 0
        while i < n:
            if flags_list[i]:
                # Find end of this run
                j = i
                while j < n and flags_list[j]:
                    j += 1
                run_len = j - i
                if run_len < min_speech_frames:
                    for k in range(i, j):
                        flags_list[k] = False
                i = j
            else:
                i += 1
        flags = torch.tensor(flags_list, dtype=torch.bool)

    return flags


def _flags_to_segments(
    flags: torch.Tensor,
    frame_duration_sec: float,
) -> List[Tuple[float, float]]:
    """Convert per-frame binary flags to (start_sec, end_sec) segment list."""
    segments: List[Tuple[float, float]] = []
    if flags.numel() == 0:
        return segments

    flags_list = flags.tolist()
    n = len(flags_list)
    in_speech = False
    start_idx = 0

    for i, is_speech in enumerate(flags_list):
        if is_speech and not in_speech:
            start_idx = i
            in_speech = True
        elif not is_speech and in_speech:
            start_sec = start_idx * frame_duration_sec
            end_sec = i * frame_duration_sec
            segments.append((start_sec, end_sec))
            in_speech = False

    if in_speech:
        start_sec = start_idx * frame_duration_sec
        end_sec = n * frame_duration_sec
        segments.append((start_sec, end_sec))

    return segments


# ---------------------------------------------------------------------------
# EnergyVAD
# ---------------------------------------------------------------------------

class EnergyVAD:
    """Fast, threshold-based voice activity detector using per-frame log-energy.

    No model, no training — works immediately on any audio.

    Args:
        config: VADConfig instance controlling frame size, threshold, etc.
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig(method="energy")

    # ------------------------------------------------------------------
    # Core frame-level API (streaming-friendly)
    # ------------------------------------------------------------------

    def process_frame(self, frame: torch.Tensor) -> bool:
        """Classify a single waveform frame as speech (True) or non-speech (False).

        Args:
            frame: 1-D tensor of ``frame_samples`` samples.

        Returns:
            True if the frame energy exceeds the configured threshold.
        """
        log_e = torch.log((frame ** 2).mean() + 1e-10).item()
        return log_e > self.config.energy_threshold

    def stream_frames(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> Generator[Tuple[int, bool], None, None]:
        """Yield ``(frame_index, is_speech)`` for each analysis frame.

        This is the streaming interface: you can process audio incrementally
        without buffering the whole recording.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Sample rate of the waveform.

        Yields:
            (frame_index, is_speech) pairs.
        """
        frame_samples = self.config.frame_samples
        frames = _frame_waveform(waveform, frame_samples)
        for i, frame in enumerate(frames):
            yield i, self.process_frame(frame)

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect speech segments in a waveform.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            List of (start_sec, end_sec) tuples, one per speech segment.
        """
        if waveform.numel() == 0:
            return []

        cfg = self.config
        frames = _frame_waveform(waveform, cfg.frame_samples)
        log_energies = _frame_log_energy(frames)
        raw_flags = log_energies > cfg.energy_threshold

        flags = _apply_hangover_and_min_duration(
            raw_flags,
            hangover_frames=cfg.hangover_frames,
            min_speech_frames=cfg.min_speech_frames,
        )
        return _flags_to_segments(flags, cfg.frame_duration_sec)

    def segment_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[torch.Tensor]:
        """Return a list of waveform slices corresponding to detected speech segments.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            List of 1-D float32 tensors, one per speech segment.
        """
        segments = self.detect(waveform, sample_rate)
        result: List[torch.Tensor] = []
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * sample_rate)
            end_idx = min(int(end_sec * sample_rate), waveform.numel())
            result.append(waveform[start_idx:end_idx])
        return result


# ---------------------------------------------------------------------------
# Lightweight Neural VAD (Silero-inspired LSTM)
# ---------------------------------------------------------------------------

class _VadLSTM(nn.Module):
    """Tiny LSTM-based binary speech/silence classifier.

    Input features: log-energy + zero-crossing-rate per frame → 2-D feature.
    Architecture: 1-layer LSTM + linear output → sigmoid.
    """

    INPUT_DIM = 2
    HIDDEN_DIM = 16

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=self.INPUT_DIM,
            hidden_size=self.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(self.HIDDEN_DIM, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: (batch, seq_len, INPUT_DIM)
            hidden: optional initial hidden state.

        Returns:
            probs: (batch, seq_len) speech probabilities.
            hidden: updated LSTM hidden state.
        """
        out, hidden = self.lstm(x, hidden)        # (batch, seq_len, HIDDEN_DIM)
        logits = self.head(out).squeeze(-1)        # (batch, seq_len)
        return torch.sigmoid(logits), hidden


def _extract_frame_features(frames: torch.Tensor) -> torch.Tensor:
    """Extract 2-D per-frame features: [log_energy, zero_crossing_rate].

    Args:
        frames: (num_frames, frame_samples)

    Returns:
        (num_frames, 2) feature tensor.
    """
    # Feature 1: log energy
    log_energy = _frame_log_energy(frames)  # (num_frames,)

    # Feature 2: zero-crossing rate
    signs = torch.sign(frames)
    zero_crossings = (signs[:, 1:] != signs[:, :-1]).float().sum(dim=1)
    zcr = zero_crossings / max(frames.shape[1] - 1, 1)  # normalise to [0, 1]

    return torch.stack([log_energy, zcr], dim=1)  # (num_frames, 2)


def _generate_training_data(
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    n_speech_segs: int = 60,
    n_silence_segs: int = 60,
    min_seg_frames: int = 5,
    max_seg_frames: int = 50,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate labelled (features, label) training pairs from synthetic audio.

    Speech is modelled as sinusoidal tones at random frequencies; silence
    is Gaussian noise at very low amplitude.

    Returns:
        features: (N, INPUT_DIM) float32 tensor.
        labels:   (N,) float32 tensor with values 0.0 (silence) or 1.0 (speech).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    def _rand_int(lo: int, hi: int) -> int:
        return int(torch.randint(lo, hi + 1, (1,), generator=rng).item())

    def _rand_float(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1, generator=rng).item()

    # Speech: moderate-amplitude tones
    for _ in range(n_speech_segs):
        n_frames = _rand_int(min_seg_frames, max_seg_frames)
        freq = _rand_float(80.0, 3400.0)  # typical speech band
        amp = _rand_float(0.3, 0.9)
        n_samples = n_frames * frame_samples
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        tone = amp * torch.sin(2.0 * math.pi * freq * t)
        # add slight noise so zero-crossing rate is non-trivial
        noise = 0.02 * torch.randn(n_samples, generator=rng)
        signal = tone + noise
        frames = signal.view(n_frames, frame_samples)
        feats = _extract_frame_features(frames)
        all_features.append(feats)
        all_labels.append(torch.ones(n_frames))

    # Silence: very low-amplitude Gaussian noise
    for _ in range(n_silence_segs):
        n_frames = _rand_int(min_seg_frames, max_seg_frames)
        amp = _rand_float(1e-5, 0.01)
        n_samples = n_frames * frame_samples
        noise = amp * torch.randn(n_samples, generator=rng)
        frames = noise.view(n_frames, frame_samples)
        feats = _extract_frame_features(frames)
        all_features.append(feats)
        all_labels.append(torch.zeros(n_frames))

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels


def _train_vad_lstm(
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    n_epochs: int = 20,
    lr: float = 1e-2,
    seed: int = 42,
) -> _VadLSTM:
    """Train a tiny VAD LSTM on synthetic data and return the fitted model."""
    torch.manual_seed(seed)
    model = _VadLSTM()
    model.train()

    features, labels = _generate_training_data(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        seed=seed,
    )

    # Treat the full dataset as a single sequence (simple but effective for tiny model)
    # shape: (1, N, INPUT_DIM)
    x = features.unsqueeze(0)
    y = labels.unsqueeze(0)  # (1, N)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        probs, _ = model(x)   # (1, N)
        loss = F.binary_cross_entropy(probs, y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


class NeuralVAD:
    """LSTM-based neural voice activity detector.

    Trains a tiny model on synthetic speech/silence examples at construction
    time (takes < 1 second). No external model files or downloads required.

    Args:
        config: VADConfig instance (method should be 'neural').
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig(method="neural")
        self._model = _train_vad_lstm(
            sample_rate=self.config.sample_rate,
            frame_duration_ms=self.config.frame_duration_ms,
        )

    def process_frame(
        self,
        frame: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
        """Score a single frame; returns (speech_prob, new_hidden_state).

        This is the streaming interface: pass the hidden state between calls.
        """
        with torch.no_grad():
            fake_frames = frame.unsqueeze(0)  # (1, frame_samples)
            feats = _extract_frame_features(fake_frames)  # (1, 2)
            x = feats.unsqueeze(0)            # (1, 1, 2)
            prob, hidden = self._model(x, hidden)
        return prob.item(), hidden

    def stream_frames(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> Generator[Tuple[int, float], None, None]:
        """Yield (frame_index, speech_probability) for each frame.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Sample rate of the waveform.

        Yields:
            (frame_index, speech_probability) pairs.
        """
        frame_samples = self.config.frame_samples
        frames = _frame_waveform(waveform, frame_samples)
        hidden = None
        for i, frame in enumerate(frames):
            prob, hidden = self.process_frame(frame, hidden)
            yield i, prob

    def detect(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect speech segments in a waveform.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            List of (start_sec, end_sec) tuples.
        """
        if waveform.numel() == 0:
            return []

        cfg = self.config
        frames = _frame_waveform(waveform, cfg.frame_samples)

        with torch.no_grad():
            feats = _extract_frame_features(frames)  # (num_frames, 2)
            x = feats.unsqueeze(0)                   # (1, num_frames, 2)
            probs, _ = self._model(x)                # (1, num_frames)
        probs = probs.squeeze(0)                     # (num_frames,)

        # Threshold at 0.5
        raw_flags = probs > 0.5

        flags = _apply_hangover_and_min_duration(
            raw_flags,
            hangover_frames=cfg.hangover_frames,
            min_speech_frames=cfg.min_speech_frames,
        )
        return _flags_to_segments(flags, cfg.frame_duration_sec)

    def segment_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[torch.Tensor]:
        """Return waveform slices for each detected speech segment."""
        segments = self.detect(waveform, sample_rate)
        result: List[torch.Tensor] = []
        for start_sec, end_sec in segments:
            start_idx = int(start_sec * sample_rate)
            end_idx = min(int(end_sec * sample_rate), waveform.numel())
            result.append(waveform[start_idx:end_idx])
        return result


# ---------------------------------------------------------------------------
# Unified VoiceActivityDetector
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """Unified VAD interface that delegates to the configured backend.

    Supports both energy-based and neural backends through a consistent API.
    Frame-by-frame streaming is available via ``stream_frames()``.

    Args:
        config: VADConfig controlling frame size, threshold, hangover, etc.
            If None, defaults to energy-based VAD with sensible defaults.
    """

    def __init__(self, config: Optional[VADConfig] = None) -> None:
        self.config = config or VADConfig(method="energy")
        if self.config.method == "energy":
            self._backend: EnergyVAD | NeuralVAD = EnergyVAD(self.config)
        elif self.config.method == "neural":
            self._backend = NeuralVAD(self.config)
        else:
            raise ValueError(f"Unknown VAD method: {self.config.method!r}")

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect speech segments in a waveform.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            List of ``(start_sec, end_sec)`` tuples, ordered by start time.
            Returns an empty list if no speech is found.
        """
        waveform = waveform.squeeze()
        if waveform.dim() != 1:
            raise ValueError(
                f"Expected a 1-D waveform tensor, got shape {tuple(waveform.shape)}"
            )
        return self._backend.detect(waveform, sample_rate)

    def segment_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> List[torch.Tensor]:
        """Return audio slices corresponding to detected speech segments.

        Args:
            waveform: 1-D float32 waveform tensor.
            sample_rate: Waveform sample rate in Hz.

        Returns:
            List of 1-D float32 tensors — one per speech segment.
        """
        waveform = waveform.squeeze()
        if waveform.dim() != 1:
            raise ValueError(
                f"Expected a 1-D waveform tensor, got shape {tuple(waveform.shape)}"
            )
        return self._backend.segment_waveform(waveform, sample_rate)

    def stream_frames(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> Generator:
        """Stream frame-by-frame VAD results for the given waveform.

        Yields:
            ``(frame_index, decision)`` where ``decision`` is a bool (energy)
            or float probability (neural).
        """
        waveform = waveform.squeeze()
        return self._backend.stream_frames(waveform, sample_rate)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> str:
        """Backend method name: 'energy' or 'neural'."""
        return self.config.method

    def __repr__(self) -> str:
        return (
            f"VoiceActivityDetector(method={self.config.method!r}, "
            f"frame_duration_ms={self.config.frame_duration_ms}, "
            f"energy_threshold={self.config.energy_threshold})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vad(
    method: str = "energy",
    frame_duration_ms: int = 20,
    energy_threshold: float = -4.0,
    min_speech_duration_ms: int = 100,
    hangover_ms: int = 200,
    sample_rate: int = 16000,
) -> VoiceActivityDetector:
    """Factory function to create a VoiceActivityDetector with the given settings.

    Args:
        method: 'energy' (default) for fast energy-based VAD, or 'neural'
            for the lightweight LSTM model.
        frame_duration_ms: Frame size in milliseconds (10, 20, or 30).
        energy_threshold: Log-energy threshold for the energy backend.
        min_speech_duration_ms: Minimum segment duration to retain.
        hangover_ms: Post-speech extension (hangover) in milliseconds.
        sample_rate: Expected sample rate of audio to be processed.

    Returns:
        A configured ``VoiceActivityDetector`` instance.

    Example::

        vad = create_vad(method='energy', frame_duration_ms=20)
        segments = vad.detect(waveform, sample_rate=16000)
    """
    config = VADConfig(
        frame_duration_ms=frame_duration_ms,
        energy_threshold=energy_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        hangover_ms=hangover_ms,
        sample_rate=sample_rate,
        method=method,
    )
    return VoiceActivityDetector(config)
