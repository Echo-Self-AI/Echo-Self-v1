"""Comprehensive tests for the end-to-end speech pipeline (Week 9).

Tests cover:
- PipelineConfig defaults
- SpeechPipeline instantiation
- process() with synthetic tone
- PipelineResult.success on valid input
- Stage latency keys
- total_latency_ms > 0
- real_time_factor > 0
- process() with silence (graceful handling)
- process() with text_input bypasses ASR
- stage_latencies['vad'] exists and is positive
- stage_latencies['nvs'] exists when nvs_enabled
- audit_event_id is non-empty string when audit_enabled
- run_latency_profile() returns correct dict structure
"""
from __future__ import annotations

import pytest
import torch

from echoself_ai.pipeline import PipelineConfig, PipelineResult, SpeechPipeline
from echoself_ai.voice.audio import generate_synthetic_tone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_pipeline() -> SpeechPipeline:
    """A fully-enabled pipeline instance shared across tests in this module."""
    cfg = PipelineConfig(
        sample_rate=16000,
        vad_enabled=True,
        asr_enabled=True,
        nvs_enabled=True,
        watermark_enabled=True,
        audit_enabled=True,
    )
    return SpeechPipeline(cfg)


@pytest.fixture(scope="module")
def tone_1s() -> torch.Tensor:
    """1-second 440 Hz synthetic tone at 16 kHz."""
    return generate_synthetic_tone(freq_hz=440.0, duration_seconds=1.0, sample_rate=16000)


@pytest.fixture(scope="module")
def tone_result(default_pipeline: SpeechPipeline, tone_1s: torch.Tensor) -> PipelineResult:
    """Cached pipeline result for a 1-second tone — avoids repeating expensive inference."""
    return default_pipeline.process(tone_1s, sample_rate=16000)


# ---------------------------------------------------------------------------
# TestPipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_default_sample_rate(self):
        cfg = PipelineConfig()
        assert cfg.sample_rate == 16000

    def test_default_vad_enabled(self):
        assert PipelineConfig().vad_enabled is True

    def test_default_asr_enabled(self):
        assert PipelineConfig().asr_enabled is True

    def test_default_nvs_enabled(self):
        assert PipelineConfig().nvs_enabled is True

    def test_default_watermark_enabled(self):
        assert PipelineConfig().watermark_enabled is True

    def test_default_audit_enabled(self):
        assert PipelineConfig().audit_enabled is True

    def test_default_timeout(self):
        assert PipelineConfig().timeout_seconds == 30.0

    def test_default_watermark_strength(self):
        assert PipelineConfig().watermark_strength == 0.01

    def test_default_speaker_verification_threshold(self):
        assert PipelineConfig().speaker_verification_threshold == 0.75

    def test_custom_values(self):
        cfg = PipelineConfig(
            sample_rate=22050,
            vad_enabled=False,
            timeout_seconds=10.0,
            watermark_strength=0.05,
        )
        assert cfg.sample_rate == 22050
        assert cfg.vad_enabled is False
        assert cfg.timeout_seconds == 10.0
        assert cfg.watermark_strength == 0.05


# ---------------------------------------------------------------------------
# TestSpeechPipelineInstantiation
# ---------------------------------------------------------------------------

class TestSpeechPipelineInstantiation:
    def test_instantiates_with_defaults(self):
        """SpeechPipeline() with no arguments must not raise."""
        pipeline = SpeechPipeline()
        assert pipeline is not None

    def test_instantiates_with_explicit_config(self):
        cfg = PipelineConfig(sample_rate=16000)
        pipeline = SpeechPipeline(cfg)
        assert pipeline.config.sample_rate == 16000

    def test_has_expected_attributes(self):
        pipeline = SpeechPipeline()
        assert hasattr(pipeline, "_vad")
        assert hasattr(pipeline, "_asr")
        assert hasattr(pipeline, "_tts")
        assert hasattr(pipeline, "_watermarker")
        assert hasattr(pipeline, "_detector")
        assert hasattr(pipeline, "_audit_logger")

    def test_all_disabled_pipeline_instantiates(self):
        cfg = PipelineConfig(
            vad_enabled=False,
            asr_enabled=False,
            nvs_enabled=False,
            watermark_enabled=False,
            audit_enabled=False,
        )
        pipeline = SpeechPipeline(cfg)
        assert pipeline is not None


# ---------------------------------------------------------------------------
# TestProcessWithSyntheticTone
# ---------------------------------------------------------------------------

class TestProcessWithSyntheticTone:
    def test_returns_pipeline_result(self, tone_result: PipelineResult):
        assert isinstance(tone_result, PipelineResult)

    def test_success_is_true(self, tone_result: PipelineResult):
        """PipelineResult.success must be True on valid input."""
        assert tone_result.success is True

    def test_job_id_is_uuid_string(self, tone_result: PipelineResult):
        import uuid
        assert uuid.UUID(tone_result.job_id)  # raises if invalid

    def test_input_duration_positive(self, tone_result: PipelineResult):
        assert tone_result.input_duration_seconds > 0.0

    def test_input_duration_approximately_one_second(self, tone_result: PipelineResult):
        assert abs(tone_result.input_duration_seconds - 1.0) < 0.01

    def test_stage_latencies_is_dict(self, tone_result: PipelineResult):
        assert isinstance(tone_result.stage_latencies, dict)

    def test_stage_latencies_contains_validate(self, tone_result: PipelineResult):
        assert "validate" in tone_result.stage_latencies

    def test_stage_latencies_contains_vad(self, tone_result: PipelineResult):
        """stage_latencies['vad'] must exist and be positive."""
        assert "vad" in tone_result.stage_latencies
        assert tone_result.stage_latencies["vad"] >= 0.0

    def test_stage_latencies_contains_asr(self, tone_result: PipelineResult):
        assert "asr" in tone_result.stage_latencies

    def test_stage_latencies_contains_nvs(self, tone_result: PipelineResult):
        """stage_latencies['nvs'] must exist when nvs_enabled=True."""
        assert "nvs" in tone_result.stage_latencies

    def test_stage_latencies_contains_watermark(self, tone_result: PipelineResult):
        assert "watermark" in tone_result.stage_latencies

    def test_stage_latencies_contains_detect(self, tone_result: PipelineResult):
        assert "detect" in tone_result.stage_latencies

    def test_stage_latencies_contains_audit(self, tone_result: PipelineResult):
        assert "audit" in tone_result.stage_latencies

    def test_total_latency_ms_positive(self, tone_result: PipelineResult):
        """PipelineResult.total_latency_ms must be > 0."""
        assert tone_result.total_latency_ms > 0.0

    def test_real_time_factor_positive(self, tone_result: PipelineResult):
        """PipelineResult.real_time_factor must be > 0."""
        assert tone_result.real_time_factor > 0.0

    def test_real_time_factor_matches_calculation(self, tone_result: PipelineResult):
        expected_rtf = tone_result.total_latency_ms / (tone_result.input_duration_seconds * 1000.0)
        assert abs(tone_result.real_time_factor - expected_rtf) < 1e-6

    def test_vad_latency_positive(self, tone_result: PipelineResult):
        """stage_latencies['vad'] exists and is positive (>= 0)."""
        assert tone_result.stage_latencies.get("vad", -1) >= 0.0

    def test_nvs_latency_positive_when_enabled(self, tone_result: PipelineResult):
        """stage_latencies['nvs'] exists when nvs_enabled=True."""
        assert "nvs" in tone_result.stage_latencies
        assert tone_result.stage_latencies["nvs"] >= 0.0

    def test_audit_event_id_non_empty_when_enabled(self, tone_result: PipelineResult):
        """audit_event_id is a non-empty string when audit_enabled=True."""
        assert tone_result.audit_event_id is not None
        assert isinstance(tone_result.audit_event_id, str)
        assert len(tone_result.audit_event_id) > 0

    def test_output_waveform_is_tensor(self, tone_result: PipelineResult):
        assert tone_result.output_waveform is not None
        assert isinstance(tone_result.output_waveform, torch.Tensor)

    def test_output_sample_rate_set(self, tone_result: PipelineResult):
        assert tone_result.output_sample_rate is not None
        assert isinstance(tone_result.output_sample_rate, int)
        assert tone_result.output_sample_rate > 0

    def test_watermark_id_is_uuid(self, tone_result: PipelineResult):
        import uuid
        assert tone_result.watermark_id is not None
        assert uuid.UUID(tone_result.watermark_id)

    def test_error_is_none_on_success(self, tone_result: PipelineResult):
        assert tone_result.error is None


# ---------------------------------------------------------------------------
# TestProcessWithSilence
# ---------------------------------------------------------------------------

class TestProcessWithSilence:
    """Silence (zeros) must be handled gracefully without crashing."""

    def test_silence_does_not_crash(self):
        pipeline = SpeechPipeline()
        silence = torch.zeros(16000, dtype=torch.float32)
        result = pipeline.process(silence, sample_rate=16000)
        assert isinstance(result, PipelineResult)

    def test_silence_has_latency_dict(self):
        pipeline = SpeechPipeline()
        silence = torch.zeros(16000, dtype=torch.float32)
        result = pipeline.process(silence, sample_rate=16000)
        assert isinstance(result.stage_latencies, dict)
        assert len(result.stage_latencies) > 0

    def test_silence_total_latency_positive(self):
        pipeline = SpeechPipeline()
        silence = torch.zeros(16000, dtype=torch.float32)
        result = pipeline.process(silence, sample_rate=16000)
        assert result.total_latency_ms > 0.0


# ---------------------------------------------------------------------------
# TestProcessWithTextInput
# ---------------------------------------------------------------------------

class TestProcessWithTextInput:
    """When text_input is provided, ASR should be skipped."""

    @pytest.fixture(scope="class")
    def text_result(self) -> PipelineResult:
        pipeline = SpeechPipeline()
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=1.0, sample_rate=16000)
        return pipeline.process(tone, sample_rate=16000, text_input="hello world")

    def test_text_input_synthesises_audio(self, text_result: PipelineResult):
        """process() with text_input='hello world' should produce output audio."""
        assert text_result.output_waveform is not None
        assert text_result.output_waveform.numel() > 0

    def test_transcript_matches_text_input(self, text_result: PipelineResult):
        """transcript should equal the provided text_input."""
        assert text_result.transcript == "hello world"

    def test_success_true_with_text_input(self, text_result: PipelineResult):
        assert text_result.success is True

    def test_stage_latencies_present_with_text_input(self, text_result: PipelineResult):
        assert "nvs" in text_result.stage_latencies
        assert text_result.stage_latencies["nvs"] >= 0.0


# ---------------------------------------------------------------------------
# TestPipelineDisabledStages
# ---------------------------------------------------------------------------

class TestPipelineDisabledStages:
    def test_no_audit_when_disabled(self):
        cfg = PipelineConfig(audit_enabled=False)
        pipeline = SpeechPipeline(cfg)
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
        result = pipeline.process(tone, sample_rate=16000, text_input="test")
        assert result.audit_event_id is None

    def test_no_watermark_when_disabled(self):
        cfg = PipelineConfig(watermark_enabled=False)
        pipeline = SpeechPipeline(cfg)
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
        result = pipeline.process(tone, sample_rate=16000, text_input="test")
        assert result.watermark_id is None

    def test_no_output_when_nvs_disabled(self):
        cfg = PipelineConfig(nvs_enabled=False, watermark_enabled=False)
        pipeline = SpeechPipeline(cfg)
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
        result = pipeline.process(tone, sample_rate=16000, text_input="test")
        assert result.output_waveform is None

    def test_nvs_latency_key_still_present_when_disabled(self):
        """The 'nvs' key should still appear in stage_latencies (0 ms) when disabled."""
        cfg = PipelineConfig(nvs_enabled=False)
        pipeline = SpeechPipeline(cfg)
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
        result = pipeline.process(tone, sample_rate=16000, text_input="test")
        assert "nvs" in result.stage_latencies


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_too_short_input_fails_gracefully(self):
        """Input shorter than MIN_SAMPLES must fail gracefully (success=False)."""
        pipeline = SpeechPipeline()
        tiny = torch.zeros(10, dtype=torch.float32)  # 10 samples < 160 minimum
        result = pipeline.process(tiny, sample_rate=16000)
        assert result.success is False
        assert result.error is not None
        assert "validate" in result.error

    def test_nan_input_fails_gracefully(self):
        """Input with NaN values must fail gracefully (success=False)."""
        pipeline = SpeechPipeline()
        nan_wave = torch.full((16000,), float("nan"), dtype=torch.float32)
        result = pipeline.process(nan_wave, sample_rate=16000)
        assert result.success is False
        assert result.error is not None

    def test_2d_input_is_squeezed(self):
        """2-D input tensors (1, N) should be handled via squeeze."""
        pipeline = SpeechPipeline()
        tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=1.0, sample_rate=16000)
        tone_2d = tone.unsqueeze(0)  # (1, N)
        result = pipeline.process(tone_2d, sample_rate=16000, text_input="test")
        assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# TestRunLatencyProfile
# ---------------------------------------------------------------------------

class TestRunLatencyProfile:
    @pytest.fixture(scope="class")
    def profile(self) -> dict:
        pipeline = SpeechPipeline()
        return pipeline.run_latency_profile(n_trials=2)

    def test_returns_dict(self, profile: dict):
        assert isinstance(profile, dict)

    def test_has_mean_key(self, profile: dict):
        """run_latency_profile() returns dict with 'mean' key."""
        assert "mean" in profile

    def test_has_std_key(self, profile: dict):
        """run_latency_profile() returns dict with 'std' key."""
        assert "std" in profile

    def test_has_min_key(self, profile: dict):
        """run_latency_profile() returns dict with 'min' key."""
        assert "min" in profile

    def test_has_max_key(self, profile: dict):
        """run_latency_profile() returns dict with 'max' key."""
        assert "max" in profile

    def test_mean_is_dict(self, profile: dict):
        assert isinstance(profile["mean"], dict)

    def test_std_is_dict(self, profile: dict):
        assert isinstance(profile["std"], dict)

    def test_min_is_dict(self, profile: dict):
        assert isinstance(profile["min"], dict)

    def test_max_is_dict(self, profile: dict):
        assert isinstance(profile["max"], dict)

    def test_has_rtf_mean(self, profile: dict):
        assert "rtf_mean" in profile

    def test_has_total_mean_ms(self, profile: dict):
        assert "total_mean_ms" in profile
        assert profile["total_mean_ms"] > 0.0

    def test_has_n_trials(self, profile: dict):
        assert "n_trials" in profile
        assert profile["n_trials"] == 2

    def test_stage_names_present_in_mean(self, profile: dict):
        expected_stages = {"validate", "vad", "asr", "nvs", "watermark", "detect", "audit"}
        assert expected_stages.issubset(set(profile["mean"].keys()))

    def test_min_le_mean_le_max(self, profile: dict):
        for stage in profile["mean"]:
            assert profile["min"][stage] <= profile["mean"][stage] <= profile["max"][stage]


# ---------------------------------------------------------------------------
# TestPipelineResultDataclass
# ---------------------------------------------------------------------------

class TestPipelineResultDataclass:
    def test_all_fields_accessible(self):
        """PipelineResult is a plain dataclass — all fields must be accessible."""
        result = PipelineResult(
            job_id="test-job",
            input_duration_seconds=1.0,
            transcript="hello",
            output_waveform=None,
            output_sample_rate=None,
            watermark_id=None,
            detection_result=None,
            audit_event_id=None,
            stage_latencies={"validate": 0.1},
            total_latency_ms=0.1,
            real_time_factor=0.0001,
            success=True,
            error=None,
        )
        assert result.job_id == "test-job"
        assert result.transcript == "hello"
        assert result.success is True
        assert result.error is None
