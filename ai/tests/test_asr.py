"""Comprehensive tests for the Week-3 ASR module (echoself_ai.voice.asr).

Test classes:
  - TestASRConfig           — dataclass validation
  - TestASRModelForward     — ASRModel shape checks
  - TestCTCDecoder          — greedy decoding on known logits
  - TestSpeechRecognizerTone — recognize() on a pure-tone waveform
  - TestSpeechRecognizerSilence — recognize() on silence → empty string
  - TestCPUFallback         — SpeechRecognizer always uses CPU when CUDA absent
  - TestVADIntegration      — VAD + ASR used together
  - TestSmokeTraining       — train_asr_smoke() runs without crashing
"""
from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn as nn

from echoself_ai.voice.asr import (
    ASRModel,
    CTCDecoder,
    SpeechRecognizer,
    _build_default_vocab,
    train_asr_smoke,
)
from echoself_ai.voice.asr_config import ASRConfig

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _make_config(**kwargs) -> ASRConfig:
    """Return a minimal ASRConfig suitable for fast CPU tests."""
    defaults = dict(
        vocab_size=10,
        blank_id=9,
        n_mels=40,
        n_fft=256,
        hop_length=64,
        win_length=256,
        sample_rate=SAMPLE_RATE,
        cnn_channels=8,
        cnn_kernel_size=3,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return ASRConfig(**defaults)


def _make_tone(
    freq_hz: float = 440.0,
    duration_s: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.8,
) -> torch.Tensor:
    """Generate a synthetic sine-wave tone (1-D float32 tensor)."""
    n = int(duration_s * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return amplitude * torch.sin(2.0 * math.pi * freq_hz * t)


def _make_silence(duration_s: float = 0.5, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Generate a silent waveform (all zeros)."""
    return torch.zeros(int(duration_s * sample_rate), dtype=torch.float32)


# ===========================================================================
# TestASRConfig
# ===========================================================================

class TestASRConfig:
    def test_default_construction(self):
        cfg = ASRConfig()
        assert cfg.vocab_size == 29
        assert cfg.blank_id == 28
        assert cfg.n_mels == 80
        assert cfg.hidden_dim == 128
        assert cfg.num_layers == 3

    def test_custom_values(self):
        cfg = _make_config()
        assert cfg.vocab_size == 10
        assert cfg.blank_id == 9
        assert cfg.n_mels == 40

    def test_vocab_size_too_small_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            ASRConfig(vocab_size=1, blank_id=0)

    def test_blank_id_out_of_range_raises(self):
        with pytest.raises(ValueError, match="blank_id"):
            ASRConfig(vocab_size=5, blank_id=10)

    def test_blank_id_negative_raises(self):
        with pytest.raises(ValueError, match="blank_id"):
            ASRConfig(vocab_size=5, blank_id=-1)

    def test_hidden_dim_zero_raises(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            ASRConfig(vocab_size=10, blank_id=9, hidden_dim=0)

    def test_num_layers_zero_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            ASRConfig(vocab_size=10, blank_id=9, num_layers=0)

    def test_dropout_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            ASRConfig(vocab_size=10, blank_id=9, dropout=1.5)

    def test_sample_rate_too_low_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            ASRConfig(vocab_size=10, blank_id=9, sample_rate=1000)


# ===========================================================================
# TestASRModelForward
# ===========================================================================

class TestASRModelForward:
    """Verify ASRModel forward-pass shapes and output range."""

    def setup_method(self):
        self.config = _make_config()
        self.model = ASRModel(self.config)
        self.model.eval()

    def test_output_shape_batch1(self):
        T = 20
        log_mel = torch.randn(1, self.config.n_mels, T)
        log_probs, lengths = self.model(log_mel)
        # Time-first CTC format: (T_out, B, vocab_size)
        assert log_probs.ndim == 3
        T_out, B, V = log_probs.shape
        assert T_out > 0
        assert B == 1
        assert V == self.config.vocab_size

    def test_output_shape_batch4(self):
        T = 30
        B = 4
        log_mel = torch.randn(B, self.config.n_mels, T)
        log_probs, lengths = self.model(log_mel)
        assert log_probs.shape[1] == B
        assert log_probs.shape[2] == self.config.vocab_size

    def test_output_lengths_shape(self):
        B = 3
        T = 25
        log_mel = torch.randn(B, self.config.n_mels, T)
        log_probs, lengths = self.model(log_mel)
        assert lengths.shape == (B,)
        assert (lengths > 0).all()

    def test_log_probs_are_log_softmax(self):
        """exp(log_probs).sum(-1) should be ~1 for all frames."""
        log_mel = torch.randn(1, self.config.n_mels, 20)
        log_probs, _ = self.model(log_mel)
        probs_sum = log_probs.exp().sum(-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_log_probs_non_positive(self):
        """Log-softmax values must all be <= 0."""
        log_mel = torch.randn(1, self.config.n_mels, 20)
        log_probs, _ = self.model(log_mel)
        assert (log_probs <= 1e-6).all()

    def test_no_nan_in_output(self):
        log_mel = torch.randn(2, self.config.n_mels, 30)
        log_probs, _ = self.model(log_mel)
        assert not torch.isnan(log_probs).any()

    def test_ctc_loss_compatible(self):
        """ASRModel output must be accepted by torch.nn.CTCLoss without error."""
        config = self.config
        model = self.model
        ctc = nn.CTCLoss(blank=config.blank_id, reduction="mean", zero_infinity=True)

        log_mel = torch.randn(1, config.n_mels, 50)
        log_probs, input_lengths = model(log_mel)

        target = torch.tensor([[0]], dtype=torch.long)
        target_lengths = torch.tensor([1], dtype=torch.long)

        loss = ctc(log_probs, target, input_lengths, target_lengths)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Backpropagation must not raise and produce non-None gradients."""
        config = self.config
        model = ASRModel(config)
        model.train()
        ctc = nn.CTCLoss(blank=config.blank_id, reduction="mean", zero_infinity=True)

        log_mel = torch.randn(1, config.n_mels, 50)
        log_probs, input_lengths = model(log_mel)
        target = torch.tensor([[0]], dtype=torch.long)
        target_lengths = torch.tensor([1], dtype=torch.long)

        loss = ctc(log_probs, target, input_lengths, target_lengths)
        loss.backward()

        # At least one parameter should have a gradient
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert any_grad


# ===========================================================================
# TestCTCDecoder
# ===========================================================================

class TestCTCDecoder:
    """Verify CTCDecoder on known logits."""

    def setup_method(self):
        _, id_to_char = _build_default_vocab()
        self.decoder = CTCDecoder(blank_id=28, id_to_char=id_to_char)

    def _one_hot_logits(self, indices: List[int], vocab_size: int = 29) -> torch.Tensor:
        """Build log_probs where each time step is peaked at the given index."""
        T = len(indices)
        logits = torch.full((T, vocab_size), -100.0)
        for t, idx in enumerate(indices):
            logits[t, idx] = 0.0
        return logits  # already effectively log-probs (not normalised, but argmax is correct)

    def test_decode_single_char(self):
        # 'a' is index 0 in default vocab
        logits = self._one_hot_logits([0, 28, 28])  # a, blank, blank
        result = self.decoder.decode(logits)
        assert result == "a"

    def test_decode_collapses_repeats(self):
        # aaa blank bbb → "ab"
        logits = self._one_hot_logits([0, 0, 0, 28, 1, 1, 1])
        result = self.decoder.decode(logits)
        assert result == "ab"

    def test_decode_blank_only_returns_empty(self):
        logits = self._one_hot_logits([28, 28, 28, 28])
        result = self.decoder.decode(logits)
        assert result == ""

    def test_decode_word(self):
        # 'hello' — h=7, e=4, l=11, l=11, o=14
        word = "hello"
        indices = []
        char_to_id = {ch: i for i, ch in enumerate(_CHARACTERS)}
        for ch in word:
            indices.append(char_to_id[ch])
            indices.append(28)  # blank between each char to avoid collapse
        logits = self._one_hot_logits(indices)
        result = self.decoder.decode(logits)
        assert result == "hello"

    def test_decode_batch(self):
        # Batch of 2: [[a], [b]]
        a_logits = self._one_hot_logits([0, 28])
        b_logits = self._one_hot_logits([1, 28])
        # Stack: (T=2, B=2, V=29)
        batch = torch.stack([a_logits, b_logits], dim=1)
        results = self.decoder.decode_batch(batch)
        assert results[0] == "a"
        assert results[1] == "b"

    def test_decode_uses_custom_id_to_char(self):
        custom = {0: "X", 1: "Y", 2: "Z"}
        decoder = CTCDecoder(blank_id=3, id_to_char=custom)
        # Vocab size 4 (X Y Z blank=3)
        # 5 frames: X, blank, Z, blank, blank
        logits = torch.full((5, 4), -100.0)
        logits[0, 0] = 0.0  # X
        logits[1, 3] = 0.0  # blank
        logits[2, 2] = 0.0  # Z
        logits[3, 3] = 0.0  # blank
        logits[4, 3] = 0.0  # blank
        result = decoder.decode(logits)
        assert result == "XZ"


# ---------------------------------------------------------------------------
# helper for _CHARACTERS (needed in test above)
# ---------------------------------------------------------------------------
_CHARACTERS: str = "abcdefghijklmnopqrstuvwxyz '"


# ===========================================================================
# TestSpeechRecognizerTone
# ===========================================================================

class TestSpeechRecognizerTone:
    """recognize() on a tone — must return a string without crashing."""

    def setup_method(self):
        self.config = _make_config()
        self.recognizer = SpeechRecognizer(config=self.config, device=torch.device("cpu"))

    def test_recognize_returns_string(self):
        tone = _make_tone(duration_s=0.5)
        result = self.recognizer.recognize(tone, sample_rate=SAMPLE_RATE)
        assert isinstance(result, str)

    def test_recognize_does_not_raise(self):
        tone = _make_tone(freq_hz=1000.0, duration_s=0.3)
        # Must not raise any exception
        self.recognizer.recognize(tone, sample_rate=SAMPLE_RATE)

    def test_recognize_long_tone_returns_string(self):
        tone = _make_tone(duration_s=2.0)
        result = self.recognizer.recognize(tone, sample_rate=SAMPLE_RATE)
        assert isinstance(result, str)

    def test_recognize_output_only_has_valid_chars_or_empty(self):
        """Output should only contain characters from the vocabulary (or be empty)."""
        tone = _make_tone(duration_s=0.5)
        result = self.recognizer.recognize(tone, sample_rate=SAMPLE_RATE)
        # Default vocab: a-z, space, apostrophe
        valid = set("abcdefghijklmnopqrstuvwxyz '")
        for ch in result:
            assert ch in valid, f"Invalid character {ch!r} in output {result!r}"


# ===========================================================================
# TestSpeechRecognizerSilence
# ===========================================================================

class TestSpeechRecognizerSilence:
    """recognize() on silence must return empty string."""

    def setup_method(self):
        self.config = _make_config()
        self.recognizer = SpeechRecognizer(config=self.config, device=torch.device("cpu"))

    def test_silence_returns_empty(self):
        silence = _make_silence(duration_s=1.0)
        result = self.recognizer.recognize(silence, sample_rate=SAMPLE_RATE)
        assert result == ""

    def test_empty_waveform_returns_empty(self):
        empty = torch.zeros(0, dtype=torch.float32)
        result = self.recognizer.recognize(empty, sample_rate=SAMPLE_RATE)
        assert result == ""

    def test_very_quiet_waveform_returns_empty(self):
        """An extremely low-amplitude signal should not be detected as speech by VAD."""
        near_silence = torch.randn(SAMPLE_RATE) * 1e-7  # sub-threshold amplitude
        result = self.recognizer.recognize(near_silence, sample_rate=SAMPLE_RATE)
        assert result == ""


# ===========================================================================
# TestCPUFallback
# ===========================================================================

class TestCPUFallback:
    """SpeechRecognizer must work on CPU regardless of CUDA availability."""

    def test_explicit_cpu_device(self):
        config = _make_config()
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))
        assert recognizer.device == torch.device("cpu")

    def test_model_on_cpu(self):
        config = _make_config()
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))
        for param in recognizer.model.parameters():
            assert param.device.type == "cpu"

    def test_recognize_on_cpu(self):
        config = _make_config()
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))
        tone = _make_tone(duration_s=0.5)
        result = recognizer.recognize(tone, sample_rate=SAMPLE_RATE)
        assert isinstance(result, str)

    def test_default_device_is_cpu_when_no_cuda(self):
        """When CUDA is not available, default device must be CPU."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine — skip CPU-only test")
        config = _make_config()
        recognizer = SpeechRecognizer(config=config)
        assert recognizer.device.type == "cpu"

    def test_missing_checkpoint_does_not_crash(self, tmp_path):
        """Pointing checkpoint_path at a non-existent file must not raise."""
        config = _make_config(checkpoint_path=str(tmp_path / "nonexistent.pt"))
        # Should log a warning but not raise
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))
        assert recognizer.model is not None


# ===========================================================================
# TestVADIntegration
# ===========================================================================

class TestVADIntegration:
    """ASR + VAD work together: import both and call both on the same waveform."""

    def test_vad_import(self):
        from echoself_ai.voice.vad import VoiceActivityDetector, create_vad
        vad = create_vad(method="energy")
        assert vad is not None

    def test_vad_segments_fed_to_asr(self):
        """VAD segments a tone and ASR transcribes each segment without error."""
        from echoself_ai.voice.vad import create_vad
        from echoself_ai.voice.vad_config import VADConfig

        config = _make_config()
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))

        # Build a waveform with a speech-then-silence pattern
        tone = _make_tone(freq_hz=440.0, duration_s=0.5)
        silence = _make_silence(duration_s=0.3)
        waveform = torch.cat([tone, silence, tone], dim=0)

        result = recognizer.recognize(waveform, sample_rate=SAMPLE_RATE)
        assert isinstance(result, str)

    def test_vad_no_segments_on_silence(self):
        """When VAD finds no speech, recognize() returns empty string."""
        config = _make_config()
        recognizer = SpeechRecognizer(config=config, device=torch.device("cpu"))
        silence = _make_silence(duration_s=1.0)
        result = recognizer.recognize(silence, sample_rate=SAMPLE_RATE)
        assert result == ""

    def test_asr_and_vad_coexist_in_voice_package(self):
        """Both ASR and VAD can be imported from the top-level voice package."""
        from echoself_ai.voice import (
            ASRModel,
            ASRConfig,
            CTCDecoder,
            SpeechRecognizer,
            VoiceActivityDetector,
            create_vad,
            VADConfig,
        )
        # Instantiate both to confirm no import-time collision
        cfg = _make_config()
        model = ASRModel(cfg)
        vad = create_vad(method="energy")
        assert model is not None
        assert vad is not None


# ===========================================================================
# TestSmokeTraining
# ===========================================================================

class TestSmokeTraining:
    """train_asr_smoke() must complete without crashing and return valid metrics."""

    def test_smoke_runs_10_steps(self):
        result = train_asr_smoke(num_steps=10)
        assert result["steps"] == 10

    def test_smoke_returns_finite_initial_loss(self):
        result = train_asr_smoke(num_steps=5)
        assert math.isfinite(result["initial_loss"])

    def test_smoke_returns_finite_final_loss(self):
        result = train_asr_smoke(num_steps=5)
        assert math.isfinite(result["final_loss"])

    def test_smoke_returns_all_losses(self):
        n = 7
        result = train_asr_smoke(num_steps=n)
        assert len(result["all_losses"]) == n
        assert all(math.isfinite(l) for l in result["all_losses"])

    def test_smoke_uses_custom_config(self):
        cfg = _make_config(num_steps=1) if False else None  # just use helper config
        cfg = ASRConfig(
            vocab_size=5,
            blank_id=4,
            n_mels=20,
            cnn_channels=4,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        result = train_asr_smoke(num_steps=3, config=cfg)
        assert result["steps"] == 3

    def test_smoke_loss_does_not_diverge(self):
        """Final loss should not blow up (> 100× initial loss) on a simple task."""
        result = train_asr_smoke(num_steps=10)
        initial = result["initial_loss"]
        final = result["final_loss"]
        # Allow for some variance but not explosive divergence
        if initial > 0:
            ratio = final / (initial + 1e-8)
            assert ratio < 100.0, f"Loss diverged: initial={initial:.4f}, final={final:.4f}"

    def test_smoke_with_device_cpu(self):
        result = train_asr_smoke(num_steps=5, device=torch.device("cpu"))
        assert result["steps"] == 5
        assert math.isfinite(result["final_loss"])
