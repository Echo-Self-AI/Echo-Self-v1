"""Comprehensive tests for the VAD module (Week 2 — Voice Activity Detection).

Tests cover:
  - detect() on a synthetic tone  → should find speech
  - detect() on silence           → should find nothing
  - detect() on mixed audio       → tone + silence + tone
  - segment_waveform() result count
  - all three frame size configurations (10 / 20 / 30 ms)
  - empty audio handling
  - min_speech_duration filtering
  - EnergyVAD standalone class
  - NeuralVAD standalone class
  - create_vad() factory function
  - VoiceActivityDetector unified interface
  - VADConfig validation
  - streaming frame-by-frame processing
"""
import math

import pytest
import torch

from echoself_ai.voice.audio import generate_synthetic_tone
from echoself_ai.voice.vad import (
    EnergyVAD,
    NeuralVAD,
    VoiceActivityDetector,
    _apply_hangover_and_min_duration,
    _flags_to_segments,
    _frame_log_energy,
    _frame_waveform,
    create_vad,
)
from echoself_ai.voice.vad_config import VADConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_MS = 20
TONE_FREQ = 440.0
TONE_AMP = 0.8
SILENCE_AMP = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tone(duration_sec: float = 1.0, amp: float = TONE_AMP) -> torch.Tensor:
    return generate_synthetic_tone(
        freq_hz=TONE_FREQ,
        duration_seconds=duration_sec,
        sample_rate=SAMPLE_RATE,
        amplitude=amp,
    )


def make_silence(duration_sec: float = 1.0) -> torch.Tensor:
    """Generate low-amplitude Gaussian noise that reads as silence."""
    n = int(duration_sec * SAMPLE_RATE)
    return torch.randn(n) * SILENCE_AMP


def make_mixed(
    speech_sec: float = 1.0,
    silence_sec: float = 1.0,
) -> torch.Tensor:
    """speech | silence | speech"""
    tone = make_tone(speech_sec)
    silence = make_silence(silence_sec)
    return torch.cat([tone, silence, tone])


# ---------------------------------------------------------------------------
# VADConfig tests
# ---------------------------------------------------------------------------

class TestVADConfig:

    def test_defaults_valid(self):
        cfg = VADConfig()
        assert cfg.frame_duration_ms == 20
        assert cfg.energy_threshold == pytest.approx(-4.0)
        assert cfg.min_speech_duration_ms == 100
        assert cfg.hangover_ms == 200
        assert cfg.sample_rate == 16000
        assert cfg.method == "energy"

    def test_frame_samples_20ms(self):
        cfg = VADConfig(frame_duration_ms=20, sample_rate=16000)
        assert cfg.frame_samples == 320

    def test_frame_samples_10ms(self):
        cfg = VADConfig(frame_duration_ms=10, sample_rate=16000)
        assert cfg.frame_samples == 160

    def test_frame_samples_30ms(self):
        cfg = VADConfig(frame_duration_ms=30, sample_rate=16000)
        assert cfg.frame_samples == 480

    def test_invalid_frame_duration(self):
        with pytest.raises(ValueError, match="frame_duration_ms"):
            VADConfig(frame_duration_ms=25)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            VADConfig(method="unknown")

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            VADConfig(sample_rate=1000)

    def test_min_speech_frames_property(self):
        cfg = VADConfig(frame_duration_ms=20, min_speech_duration_ms=100)
        assert cfg.min_speech_frames == 5  # 100ms / 20ms = 5

    def test_hangover_frames_property(self):
        cfg = VADConfig(frame_duration_ms=20, hangover_ms=200)
        assert cfg.hangover_frames == 10  # 200ms / 20ms = 10


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:

    def test_frame_waveform_exact_multiple(self):
        waveform = torch.ones(640)
        frames = _frame_waveform(waveform, 320)
        assert frames.shape == (2, 320)

    def test_frame_waveform_pads_last(self):
        waveform = torch.ones(500)
        frames = _frame_waveform(waveform, 320)
        # 500 / 320 → 2 frames after padding to 640
        assert frames.shape == (2, 320)

    def test_frame_waveform_empty(self):
        waveform = torch.tensor([])
        frames = _frame_waveform(waveform, 320)
        assert frames.shape == (0, 320)

    def test_frame_log_energy_tone_vs_silence(self):
        tone_frames = make_tone(0.5).view(-1, 320)
        silence_frames = make_silence(0.5).view(-1, 320)
        tone_energy = _frame_log_energy(tone_frames).mean().item()
        silence_energy = _frame_log_energy(silence_frames).mean().item()
        assert tone_energy > silence_energy

    def test_flags_to_segments_basic(self):
        flags = torch.tensor([False, True, True, False, False])
        segs = _flags_to_segments(flags, frame_duration_sec=0.02)
        assert len(segs) == 1
        start, end = segs[0]
        assert start == pytest.approx(0.02)
        assert end == pytest.approx(0.06)

    def test_flags_to_segments_empty(self):
        flags = torch.tensor([])
        segs = _flags_to_segments(flags, frame_duration_sec=0.02)
        assert segs == []

    def test_flags_to_segments_all_speech(self):
        flags = torch.ones(5, dtype=torch.bool)
        segs = _flags_to_segments(flags, frame_duration_sec=0.02)
        assert len(segs) == 1
        assert segs[0][0] == pytest.approx(0.0)
        assert segs[0][1] == pytest.approx(0.10)

    def test_hangover_extends_segment(self):
        flags = torch.tensor([False, True, False, False, False, False], dtype=torch.bool)
        result = _apply_hangover_and_min_duration(flags, hangover_frames=2, min_speech_frames=1)
        # Frame 1 is speech; hangover should extend to frames 2 and 3
        assert result[1].item() is True
        assert result[2].item() is True
        assert result[3].item() is True
        assert result[0].item() is False
        assert result[4].item() is False

    def test_min_duration_removes_short_segments(self):
        # 1 frame of speech should be removed when min=3
        flags = torch.tensor([False, True, False, False, False], dtype=torch.bool)
        result = _apply_hangover_and_min_duration(flags, hangover_frames=0, min_speech_frames=3)
        assert not result.any()

    def test_min_duration_keeps_long_segments(self):
        flags = torch.tensor([False, True, True, True, True, False], dtype=torch.bool)
        result = _apply_hangover_and_min_duration(flags, hangover_frames=0, min_speech_frames=3)
        assert result[1:5].all()


# ---------------------------------------------------------------------------
# EnergyVAD tests
# ---------------------------------------------------------------------------

class TestEnergyVAD:

    def _make_vad(self, **kwargs) -> EnergyVAD:
        cfg = VADConfig(
            method="energy",
            frame_duration_ms=FRAME_MS,
            hangover_ms=0,
            min_speech_duration_ms=0,
            **kwargs,
        )
        return EnergyVAD(cfg)

    def test_detect_tone_finds_speech(self):
        vad = self._make_vad()
        tone = make_tone(1.0)
        segments = vad.detect(tone, SAMPLE_RATE)
        assert len(segments) >= 1
        total_coverage = sum(e - s for s, e in segments)
        assert total_coverage > 0.5  # at least half the second is speech

    def test_detect_silence_finds_nothing(self):
        vad = self._make_vad()
        silence = make_silence(1.0)
        segments = vad.detect(silence, SAMPLE_RATE)
        assert len(segments) == 0

    def test_detect_mixed_finds_two_segments(self):
        vad = self._make_vad()
        mixed = make_mixed(speech_sec=1.0, silence_sec=1.0)
        segments = vad.detect(mixed, SAMPLE_RATE)
        assert len(segments) == 2

    def test_detect_empty_audio(self):
        vad = self._make_vad()
        segments = vad.detect(torch.tensor([]), SAMPLE_RATE)
        assert segments == []

    def test_segment_waveform_returns_two_tensors(self):
        vad = self._make_vad()
        mixed = make_mixed(speech_sec=1.0, silence_sec=1.0)
        tensors = vad.segment_waveform(mixed, SAMPLE_RATE)
        assert len(tensors) == 2
        for t in tensors:
            assert t.dim() == 1
            assert t.numel() > 0

    def test_frame_duration_10ms(self):
        vad = EnergyVAD(VADConfig(method="energy", frame_duration_ms=10, hangover_ms=0, min_speech_duration_ms=0))
        tone = make_tone(0.5)
        segments = vad.detect(tone, SAMPLE_RATE)
        assert len(segments) >= 1

    def test_frame_duration_30ms(self):
        vad = EnergyVAD(VADConfig(method="energy", frame_duration_ms=30, hangover_ms=0, min_speech_duration_ms=0))
        tone = make_tone(0.5)
        segments = vad.detect(tone, SAMPLE_RATE)
        assert len(segments) >= 1

    def test_min_speech_duration_filters_short_bursts(self):
        # Very short tone (100 ms) with min_speech=500 ms → should be filtered out
        vad = EnergyVAD(VADConfig(
            method="energy",
            frame_duration_ms=20,
            hangover_ms=0,
            min_speech_duration_ms=500,
        ))
        short_tone = make_tone(0.1)  # 100 ms
        segments = vad.detect(short_tone, SAMPLE_RATE)
        assert len(segments) == 0

    def test_min_speech_duration_keeps_long_speech(self):
        vad = EnergyVAD(VADConfig(
            method="energy",
            frame_duration_ms=20,
            hangover_ms=0,
            min_speech_duration_ms=100,
        ))
        long_tone = make_tone(1.0)
        segments = vad.detect(long_tone, SAMPLE_RATE)
        assert len(segments) >= 1

    def test_process_frame_tone(self):
        vad = self._make_vad()
        frame = make_tone(0.02)  # exactly one 20ms frame
        result = vad.process_frame(frame)
        assert result is True

    def test_process_frame_silence(self):
        vad = self._make_vad()
        frame = make_silence(0.02)
        result = vad.process_frame(frame)
        assert result is False

    def test_stream_frames_yields_all_frames(self):
        vad = self._make_vad()
        tone = make_tone(0.5)
        frame_samples = VADConfig(frame_duration_ms=FRAME_MS).frame_samples
        expected_frames = math.ceil(tone.numel() / frame_samples)
        yielded = list(vad.stream_frames(tone, SAMPLE_RATE))
        assert len(yielded) == expected_frames
        for i, (idx, decision) in enumerate(yielded):
            assert idx == i
            assert isinstance(decision, bool)

    def test_segment_order_is_chronological(self):
        vad = self._make_vad()
        mixed = make_mixed(speech_sec=0.5, silence_sec=0.5)
        segments = vad.detect(mixed, SAMPLE_RATE)
        for i in range(len(segments) - 1):
            assert segments[i][1] <= segments[i + 1][0]


# ---------------------------------------------------------------------------
# NeuralVAD tests
# ---------------------------------------------------------------------------

class TestNeuralVAD:

    def _make_vad(self, **kwargs) -> NeuralVAD:
        cfg = VADConfig(
            method="neural",
            frame_duration_ms=FRAME_MS,
            hangover_ms=0,
            min_speech_duration_ms=0,
            **kwargs,
        )
        return NeuralVAD(cfg)

    def test_detect_tone_finds_speech(self):
        vad = self._make_vad()
        tone = make_tone(1.0)
        segments = vad.detect(tone, SAMPLE_RATE)
        assert len(segments) >= 1

    def test_detect_silence_finds_nothing(self):
        vad = self._make_vad()
        silence = make_silence(1.0)
        segments = vad.detect(silence, SAMPLE_RATE)
        assert len(segments) == 0

    def test_detect_empty_audio(self):
        vad = self._make_vad()
        segments = vad.detect(torch.tensor([]), SAMPLE_RATE)
        assert segments == []

    def test_segment_waveform_mixed(self):
        vad = self._make_vad()
        mixed = make_mixed(speech_sec=1.0, silence_sec=1.0)
        tensors = vad.segment_waveform(mixed, SAMPLE_RATE)
        assert len(tensors) >= 1
        for t in tensors:
            assert t.dim() == 1

    def test_stream_frames_yields_probabilities(self):
        vad = self._make_vad()
        tone = make_tone(0.1)
        yielded = list(vad.stream_frames(tone, SAMPLE_RATE))
        assert len(yielded) > 0
        for idx, prob in yielded:
            assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# VoiceActivityDetector unified interface tests
# ---------------------------------------------------------------------------

class TestVoiceActivityDetector:

    def test_default_uses_energy(self):
        vad = VoiceActivityDetector()
        assert vad.method == "energy"

    def test_neural_method(self):
        cfg = VADConfig(method="neural")
        vad = VoiceActivityDetector(cfg)
        assert vad.method == "neural"

    def test_detect_tone_energy(self):
        vad = VoiceActivityDetector(VADConfig(
            method="energy", hangover_ms=0, min_speech_duration_ms=0
        ))
        tone = make_tone(1.0)
        segs = vad.detect(tone, SAMPLE_RATE)
        assert len(segs) >= 1

    def test_detect_silence_energy(self):
        vad = VoiceActivityDetector(VADConfig(
            method="energy", hangover_ms=0, min_speech_duration_ms=0
        ))
        silence = make_silence(1.0)
        segs = vad.detect(silence, SAMPLE_RATE)
        assert len(segs) == 0

    def test_detect_mixed_energy(self):
        vad = VoiceActivityDetector(VADConfig(
            method="energy", hangover_ms=0, min_speech_duration_ms=0
        ))
        mixed = make_mixed(1.0, 1.0)
        segs = vad.detect(mixed, SAMPLE_RATE)
        assert len(segs) == 2

    def test_detect_empty_audio(self):
        vad = VoiceActivityDetector()
        segs = vad.detect(torch.tensor([]), SAMPLE_RATE)
        assert segs == []

    def test_segment_waveform_count(self):
        vad = VoiceActivityDetector(VADConfig(
            method="energy", hangover_ms=0, min_speech_duration_ms=0
        ))
        mixed = make_mixed(1.0, 1.0)
        tensors = vad.segment_waveform(mixed, SAMPLE_RATE)
        assert len(tensors) == 2

    def test_segment_waveform_tensors_non_empty(self):
        vad = VoiceActivityDetector(VADConfig(
            method="energy", hangover_ms=0, min_speech_duration_ms=0
        ))
        tone = make_tone(1.0)
        tensors = vad.segment_waveform(tone, SAMPLE_RATE)
        assert len(tensors) >= 1
        for t in tensors:
            assert t.numel() > 0

    def test_repr_contains_method(self):
        vad = VoiceActivityDetector()
        assert "energy" in repr(vad)

    def test_stream_frames(self):
        vad = VoiceActivityDetector(VADConfig(method="energy"))
        tone = make_tone(0.1)
        frames = list(vad.stream_frames(tone, SAMPLE_RATE))
        assert len(frames) > 0

    def test_frame_duration_configs(self):
        """Test all three valid frame duration values."""
        for ms in (10, 20, 30):
            cfg = VADConfig(
                method="energy",
                frame_duration_ms=ms,
                hangover_ms=0,
                min_speech_duration_ms=0,
            )
            vad = VoiceActivityDetector(cfg)
            tone = make_tone(0.5)
            segs = vad.detect(tone, SAMPLE_RATE)
            assert len(segs) >= 1, f"frame_duration_ms={ms} failed to detect speech"

    def test_min_speech_duration_filters(self):
        """Very short speech should be suppressed by min_speech_duration_ms."""
        vad = VoiceActivityDetector(VADConfig(
            method="energy",
            frame_duration_ms=20,
            hangover_ms=0,
            min_speech_duration_ms=500,
        ))
        short_tone = make_tone(0.05)  # 50 ms — well below 500 ms threshold
        segs = vad.detect(short_tone, SAMPLE_RATE)
        assert len(segs) == 0

    def test_hangover_merges_close_segments(self):
        """With long hangover, two close speech segments should merge into one."""
        # Create: 200ms tone, 100ms silence, 200ms tone
        tone1 = make_tone(0.2)
        gap = make_silence(0.1)
        tone2 = make_tone(0.2)
        mixed = torch.cat([tone1, gap, tone2])

        # Without hangover → 2 segments
        vad_no_hangover = VoiceActivityDetector(VADConfig(
            method="energy",
            frame_duration_ms=20,
            hangover_ms=0,
            min_speech_duration_ms=0,
        ))
        segs_no_ho = vad_no_hangover.detect(mixed, SAMPLE_RATE)

        # With 300ms hangover → segments should merge
        vad_with_hangover = VoiceActivityDetector(VADConfig(
            method="energy",
            frame_duration_ms=20,
            hangover_ms=300,
            min_speech_duration_ms=0,
        ))
        segs_with_ho = vad_with_hangover.detect(mixed, SAMPLE_RATE)

        # Hangover version should have fewer (merged) segments
        assert len(segs_with_ho) <= len(segs_no_ho)


# ---------------------------------------------------------------------------
# create_vad factory tests
# ---------------------------------------------------------------------------

class TestCreateVad:

    def test_default_creates_energy_vad(self):
        vad = create_vad()
        assert vad.method == "energy"
        assert isinstance(vad, VoiceActivityDetector)

    def test_neural_factory(self):
        vad = create_vad(method="neural")
        assert vad.method == "neural"

    def test_custom_frame_duration(self):
        vad = create_vad(frame_duration_ms=10)
        assert vad.config.frame_duration_ms == 10

    def test_custom_threshold(self):
        vad = create_vad(energy_threshold=-6.0)
        assert vad.config.energy_threshold == pytest.approx(-6.0)

    def test_detect_runs_with_factory_default(self):
        vad = create_vad()
        tone = make_tone(1.0)
        segs = vad.detect(tone, SAMPLE_RATE)
        assert len(segs) >= 1

    def test_silence_with_factory_default(self):
        vad = create_vad()
        silence = make_silence(1.0)
        segs = vad.detect(silence, SAMPLE_RATE)
        assert len(segs) == 0

    def test_mixed_with_factory(self):
        vad = create_vad(hangover_ms=0, min_speech_duration_ms=0)
        mixed = make_mixed(1.0, 1.0)
        segs = vad.detect(mixed, SAMPLE_RATE)
        assert len(segs) == 2
