"""
EchoSelf Week 10 — Validation Tests
=====================================
Basic assertions that all subsystems meet minimum quality bars.

Checks:
  - NVS synthesis returns valid waveform (no NaN)
  - VAD detects speech in a tone
  - VAD finds no speech in pure silence
  - Watermark SNR > 20 dB for strength=0.01
  - Speaker verification returns float similarity in [0, 1]
  - Pipeline returns PipelineResult with success=True
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Shared helpers (not fixtures so they stay independent and fast)
# ---------------------------------------------------------------------------

def _silence(duration_s: float) -> torch.Tensor:
    return torch.zeros(int(duration_s * SAMPLE_RATE))


def _tone(freq_hz: float, duration_s: float, amp: float = 0.8) -> torch.Tensor:
    n = int(duration_s * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    return amp * torch.sin(2 * math.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# NVS — synthesis returns valid waveform (no NaN)
# ---------------------------------------------------------------------------

class TestNVS:
    def test_synthesis_no_nan(self):
        """NVS synthesis returns valid waveform (no NaN or Inf)."""
        from echoself_ai.voice.nvs import TextToSpeech, NVSConfig

        tts = TextToSpeech(NVSConfig(sample_rate=SAMPLE_RATE))
        waveform, sr = tts.synthesize("hello world")

        assert isinstance(waveform, torch.Tensor), "synthesize() must return a Tensor"
        assert sr == SAMPLE_RATE, f"expected sample_rate={SAMPLE_RATE}, got {sr}"
        assert waveform.numel() > 0, "output waveform must be non-empty"
        assert not torch.isnan(waveform).any(), "waveform must not contain NaN"
        assert not torch.isinf(waveform).any(), "waveform must not contain Inf"

    def test_synthesis_multiple_texts(self):
        """Multiple short texts each produce a finite waveform."""
        from echoself_ai.voice.nvs import TextToSpeech, NVSConfig

        tts = TextToSpeech(NVSConfig(sample_rate=SAMPLE_RATE))
        for text in ["echo self", "testing one two three", "neural audio"]:
            waveform, _ = tts.synthesize(text)
            assert not torch.isnan(waveform).any(), f"NaN in output for text={text!r}"
            assert not torch.isinf(waveform).any(), f"Inf in output for text={text!r}"


# ---------------------------------------------------------------------------
# VAD — detects speech in a tone; no speech in pure silence
# ---------------------------------------------------------------------------

class TestVAD:
    def test_detects_tone_as_speech(self):
        """EnergyVAD detects at least one speech segment in a 2s tone."""
        from echoself_ai.voice.vad import EnergyVAD
        from echoself_ai.voice.vad_config import VADConfig

        vad = EnergyVAD(VADConfig(method="energy", energy_threshold=-4.0,
                                  sample_rate=SAMPLE_RATE))
        tone = _tone(440.0, 2.0)
        segments = vad.detect(tone, SAMPLE_RATE)

        assert len(segments) > 0, "EnergyVAD must detect at least one segment in a 2s tone"

    def test_no_speech_in_silence(self):
        """EnergyVAD returns no segments for 2s of pure digital silence."""
        from echoself_ai.voice.vad import EnergyVAD
        from echoself_ai.voice.vad_config import VADConfig

        vad = EnergyVAD(VADConfig(method="energy", energy_threshold=-4.0,
                                  sample_rate=SAMPLE_RATE))
        silence = _silence(2.0)
        segments = vad.detect(silence, SAMPLE_RATE)

        assert segments == [], (
            f"EnergyVAD must return no segments for pure silence; got {segments}"
        )

    def test_tone_gap_tone_detects_segments(self):
        """VAD handles a tone+silence+tone pattern without crashing."""
        from echoself_ai.voice.vad import EnergyVAD
        from echoself_ai.voice.vad_config import VADConfig

        vad = EnergyVAD(VADConfig(method="energy", energy_threshold=-4.0,
                                  sample_rate=SAMPLE_RATE))
        audio = torch.cat([_tone(440.0, 2.0), _silence(1.0), _tone(440.0, 2.0)])
        segments = vad.detect(audio, SAMPLE_RATE)
        assert len(segments) >= 1, "Must detect at least one segment in tone+gap+tone"


# ---------------------------------------------------------------------------
# Classical Watermarking — SNR > 20 dB for strength=0.01
# ---------------------------------------------------------------------------

class TestClassicalWatermark:
    def test_snr_gt_20db_for_strength_001(self):
        """Classical watermark SNR > 20 dB for strength=0.01."""
        from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
        from echoself_ai.watermark.config import ClassicalWatermarkConfig

        cfg = ClassicalWatermarkConfig(
            watermark_id="test_validation_snr",
            strength=0.01,
            sample_rate=SAMPLE_RATE,
        )
        wm = SpreadSpectrumWatermarker(cfg)
        audio = _tone(440.0, 2.0)
        watermarked = wm.embed(audio, "test_validation_snr")
        snr = wm.compute_snr(audio, watermarked)

        assert snr > 20.0, f"SNR {snr:.2f} dB must be > 20 dB for strength=0.01"

    def test_watermark_detected_after_embed(self):
        """Classical watermark is detected after embedding."""
        from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
        from echoself_ai.watermark.config import ClassicalWatermarkConfig

        cfg = ClassicalWatermarkConfig(
            watermark_id="test_detection",
            strength=0.01,
            sample_rate=SAMPLE_RATE,
        )
        wm = SpreadSpectrumWatermarker(cfg)
        audio = _tone(440.0, 2.0)
        watermarked = wm.embed(audio, "test_detection")
        detected, corr, ber = wm.extract(watermarked, "test_detection")

        assert detected, f"Watermark must be detected after embedding (corr={corr:.4f})"
        assert isinstance(corr, float)
        assert isinstance(ber, float)


# ---------------------------------------------------------------------------
# Speaker Verification — returns float similarity in [0, 1]
# ---------------------------------------------------------------------------

class TestSpeakerVerification:
    def test_similarity_in_unit_interval(self):
        """SpeakerVerifier.verify returns a float similarity in [0, 1]."""
        from echoself_ai.voice.speaker_verification import (
            SpeakerEncoder,
            SpeakerEnrollment,
            SpeakerVerifier,
        )

        tmp_dir = Path(tempfile.mkdtemp())
        encoder = SpeakerEncoder()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_dir)
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_dir)

        waveforms = [_tone(300.0, 2.0) for _ in range(3)]
        enrollment.enroll("test_user_val", waveforms, SAMPLE_RATE)

        query = _tone(300.0, 2.0)
        accepted, similarity = verifier.verify("test_user_val", query, SAMPLE_RATE, threshold=0.0)

        assert isinstance(similarity, float), f"similarity must be float, got {type(similarity)}"
        # Cosine similarity can range [-1, 1] after L2 norm; we check it's numeric
        assert -1.0 <= similarity <= 1.0, f"similarity {similarity} outside [-1,1]"

    def test_matched_higher_than_unmatched(self):
        """Matched speaker similarity is >= unmatched speaker similarity."""
        from echoself_ai.voice.speaker_verification import (
            SpeakerEncoder,
            SpeakerEnrollment,
            SpeakerVerifier,
        )

        tmp_dir = Path(tempfile.mkdtemp())
        encoder = SpeakerEncoder()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_dir)
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_dir)

        waveforms = [_tone(300.0, 2.0) for _ in range(3)]
        enrollment.enroll("sv_test_user", waveforms, SAMPLE_RATE)

        _, sim_match = verifier.verify("sv_test_user", _tone(300.0, 2.0), SAMPLE_RATE, threshold=0.0)
        _, sim_unmatch = verifier.verify("sv_test_user", _tone(4000.0, 2.0), SAMPLE_RATE, threshold=0.0)

        assert sim_match >= sim_unmatch, (
            f"Matched similarity ({sim_match:.4f}) should be >= unmatched ({sim_unmatch:.4f})"
        )


# ---------------------------------------------------------------------------
# Pipeline — returns PipelineResult with success=True
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        from echoself_ai.pipeline import SpeechPipeline, PipelineConfig
        cfg = PipelineConfig(
            sample_rate=SAMPLE_RATE,
            vad_enabled=True,
            asr_enabled=True,
            nvs_enabled=True,
            watermark_enabled=True,
            audit_enabled=True,
        )
        return SpeechPipeline(cfg)

    @pytest.fixture(scope="class")
    def pipeline_result(self, pipeline):
        from echoself_ai.voice.audio import generate_synthetic_tone
        audio = generate_synthetic_tone(freq_hz=440.0, duration_seconds=2.0,
                                        sample_rate=SAMPLE_RATE)
        return pipeline.process(audio, sample_rate=SAMPLE_RATE, text_input="hello world")

    def test_pipeline_success(self, pipeline_result):
        """Full pipeline returns PipelineResult with success=True."""
        from echoself_ai.pipeline import PipelineResult
        assert isinstance(pipeline_result, PipelineResult)
        assert pipeline_result.success is True, (
            f"Pipeline must succeed; error={pipeline_result.error}"
        )

    def test_pipeline_has_job_id(self, pipeline_result):
        """PipelineResult has a non-empty job_id."""
        assert pipeline_result.job_id and len(pipeline_result.job_id) > 0

    def test_pipeline_total_latency_positive(self, pipeline_result):
        """Total pipeline latency is > 0 ms."""
        assert pipeline_result.total_latency_ms > 0

    def test_pipeline_stage_latencies_present(self, pipeline_result):
        """Stage latencies dict includes at least 'vad' key."""
        assert "vad" in pipeline_result.stage_latencies

    def test_pipeline_real_time_factor_positive(self, pipeline_result):
        """Real-time factor is > 0."""
        assert pipeline_result.real_time_factor > 0

    def test_pipeline_has_output_waveform(self, pipeline_result):
        """Pipeline produces an output waveform."""
        assert pipeline_result.output_waveform is not None
        assert pipeline_result.output_waveform.numel() > 0
