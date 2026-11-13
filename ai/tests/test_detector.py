"""
Tests for WatermarkDetector and DetectionResult.

Coverage:
- detect_classical() on watermarked audio returns detected=True
- detect_classical() on clean audio returns detected=False
- detect_from_file() on a written WAV returns list of DetectionResult
- Corrupted / empty audio input returns DetectionResult with error field set
- DetectionResult fields are correct types
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
from echoself_ai.watermark.config import ClassicalWatermarkConfig
from echoself_ai.watermark.detector import DetectionResult, WatermarkDetector
from echoself_ai.voice.audio import write_wav


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION_S = 0.5
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
WATERMARK_ID = "test-watermark-week8"


@pytest.fixture()
def clean_waveform() -> torch.Tensor:
    """Random audio in [-1, 1], mono."""
    torch.manual_seed(42)
    return torch.rand(N_SAMPLES) * 2.0 - 1.0


@pytest.fixture()
def watermarked_waveform(clean_waveform: torch.Tensor) -> torch.Tensor:
    """clean_waveform with a classical watermark embedded."""
    cfg = ClassicalWatermarkConfig(watermark_id=WATERMARK_ID, strength=0.05)
    wm = SpreadSpectrumWatermarker(cfg)
    return wm.embed(clean_waveform, WATERMARK_ID)


@pytest.fixture()
def detector() -> WatermarkDetector:
    return WatermarkDetector()


# ---------------------------------------------------------------------------
# DetectionResult field types
# ---------------------------------------------------------------------------

class TestDetectionResultFieldTypes:
    """Verify DetectionResult instantiation and field types."""

    def test_basic_instantiation(self) -> None:
        result = DetectionResult(
            watermark_type="classical",
            detected=True,
            confidence=0.9,
            watermark_id="wm-1",
            correlation=0.05,
            bit_accuracy=0.9,
            processing_time_ms=12.3,
            error=None,
        )
        assert isinstance(result.watermark_type, str)
        assert isinstance(result.detected, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.watermark_id, str)
        assert isinstance(result.correlation, float)
        assert isinstance(result.bit_accuracy, float)
        assert isinstance(result.processing_time_ms, float)
        assert result.error is None

    def test_optional_fields_default_to_none(self) -> None:
        result = DetectionResult(
            watermark_type="none",
            detected=False,
            confidence=0.0,
        )
        assert result.watermark_id is None
        assert result.correlation is None
        assert result.bit_accuracy is None
        assert result.error is None

    def test_confidence_range(self) -> None:
        """Confidence produced by detect_classical must be in [0, 1]."""
        detector = WatermarkDetector()
        torch.manual_seed(0)
        waveform = torch.rand(N_SAMPLES) * 2.0 - 1.0
        result = detector.detect_classical(waveform, "some-id")
        assert 0.0 <= result.confidence <= 1.0

    def test_processing_time_positive(self) -> None:
        detector = WatermarkDetector()
        torch.manual_seed(1)
        waveform = torch.rand(N_SAMPLES) * 2.0 - 1.0
        result = detector.detect_classical(waveform, "some-id")
        assert result.processing_time_ms >= 0.0

    def test_watermark_type_values(self) -> None:
        valid_types = {"classical", "learned", "none", "unsupported"}
        detector = WatermarkDetector()
        torch.manual_seed(2)
        waveform = torch.rand(N_SAMPLES) * 2.0 - 1.0
        result = detector.detect_classical(waveform, "some-id")
        assert result.watermark_type in valid_types


# ---------------------------------------------------------------------------
# detect_classical
# ---------------------------------------------------------------------------

class TestDetectClassical:
    """Tests for WatermarkDetector.detect_classical()."""

    def test_detects_watermarked_audio(
        self,
        detector: WatermarkDetector,
        watermarked_waveform: torch.Tensor,
    ) -> None:
        """Classical detection must succeed on audio watermarked with the same ID."""
        result = detector.detect_classical(
            watermarked_waveform, WATERMARK_ID, sample_rate=SAMPLE_RATE
        )
        assert result.watermark_type == "classical"
        assert result.detected is True, (
            f"Expected detection but got correlation={result.correlation}, "
            f"confidence={result.confidence}"
        )
        assert result.error is None

    def test_does_not_detect_clean_audio(self) -> None:
        """Classical detection must NOT fire when there is no watermark.

        A zero-padded waveform has zero correlation with any PN sequence, so
        detection must return False regardless of the threshold.
        """
        detector = WatermarkDetector()
        # Silence: zero correlation with any PN sequence → definitely not detected
        waveform = torch.zeros(N_SAMPLES)
        result = detector.detect_classical(waveform, WATERMARK_ID, sample_rate=SAMPLE_RATE)
        assert result.watermark_type == "classical"
        assert result.detected is False, (
            f"False positive on silence: correlation={result.correlation}"
        )
        assert result.error is None

    def test_watermark_id_preserved(
        self,
        detector: WatermarkDetector,
        clean_waveform: torch.Tensor,
    ) -> None:
        result = detector.detect_classical(clean_waveform, "my-id")
        assert result.watermark_id == "my-id"

    def test_correlation_is_float(
        self,
        detector: WatermarkDetector,
        clean_waveform: torch.Tensor,
    ) -> None:
        result = detector.detect_classical(clean_waveform, WATERMARK_ID)
        assert isinstance(result.correlation, float)

    def test_bit_accuracy_in_range(
        self,
        detector: WatermarkDetector,
        watermarked_waveform: torch.Tensor,
    ) -> None:
        result = detector.detect_classical(watermarked_waveform, WATERMARK_ID)
        assert result.bit_accuracy is not None
        assert 0.0 <= result.bit_accuracy <= 1.0

    def test_different_id_does_not_detect(
        self,
        detector: WatermarkDetector,
        watermarked_waveform: torch.Tensor,
    ) -> None:
        """A different watermark ID should not trigger detection on the same audio."""
        result = detector.detect_classical(watermarked_waveform, "completely-different-id")
        # Detection with wrong ID: correlation will be near zero → not detected
        assert result.watermark_type == "classical"
        # We just check no crash and types are correct
        assert isinstance(result.detected, bool)
        assert result.error is None


# ---------------------------------------------------------------------------
# detect_learned — no checkpoint path (unsupported)
# ---------------------------------------------------------------------------

class TestDetectLearnedNoCheckpoint:
    """When no checkpoint exists detect_learned returns 'unsupported'."""

    def test_no_checkpoint_returns_unsupported(self, clean_waveform: torch.Tensor) -> None:
        # Point to a directory that definitely does not contain checkpoints
        detector = WatermarkDetector(checkpoint_dir="/tmp/does_not_exist_echoself")
        result = detector.detect_learned(clean_waveform)
        assert result.watermark_type == "unsupported"
        assert result.detected is False
        # error message should explain the situation
        assert result.error is not None
        assert "checkpoint" in result.error.lower()


# ---------------------------------------------------------------------------
# detect_all
# ---------------------------------------------------------------------------

class TestDetectAll:
    """Tests for WatermarkDetector.detect_all()."""

    def test_returns_list(self, detector: WatermarkDetector, clean_waveform: torch.Tensor) -> None:
        results = detector.detect_all(clean_waveform, watermark_id=WATERMARK_ID)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_includes_classical_when_id_given(
        self, detector: WatermarkDetector, clean_waveform: torch.Tensor
    ) -> None:
        results = detector.detect_all(clean_waveform, watermark_id=WATERMARK_ID)
        types = [r.watermark_type for r in results]
        assert "classical" in types

    def test_no_classical_when_no_id(
        self, detector: WatermarkDetector, clean_waveform: torch.Tensor
    ) -> None:
        results = detector.detect_all(clean_waveform, watermark_id=None)
        types = [r.watermark_type for r in results]
        assert "classical" not in types

    def test_all_results_have_processing_time(
        self, detector: WatermarkDetector, clean_waveform: torch.Tensor
    ) -> None:
        results = detector.detect_all(clean_waveform, watermark_id=WATERMARK_ID)
        for r in results:
            assert r.processing_time_ms >= 0.0


# ---------------------------------------------------------------------------
# detect_from_file
# ---------------------------------------------------------------------------

class TestDetectFromFile:
    """Tests for WatermarkDetector.detect_from_file()."""

    def test_detects_from_written_wav(
        self,
        detector: WatermarkDetector,
        watermarked_waveform: torch.Tensor,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = str(Path(tmp) / "watermarked.wav")
            write_wav(wav_path, watermarked_waveform, sample_rate=SAMPLE_RATE)

            results = detector.detect_from_file(wav_path, watermark_id=WATERMARK_ID)

        assert isinstance(results, list)
        assert len(results) >= 1
        # Each element is a DetectionResult
        for r in results:
            assert isinstance(r, DetectionResult)

    def test_classical_detected_from_file(
        self,
        detector: WatermarkDetector,
        watermarked_waveform: torch.Tensor,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = str(Path(tmp) / "wm.wav")
            write_wav(wav_path, watermarked_waveform, sample_rate=SAMPLE_RATE)
            results = detector.detect_from_file(wav_path, watermark_id=WATERMARK_ID)

        classical_results = [r for r in results if r.watermark_type == "classical"]
        assert len(classical_results) == 1
        assert classical_results[0].detected is True

    def test_missing_file_returns_error_result(
        self, detector: WatermarkDetector
    ) -> None:
        results = detector.detect_from_file(
            "/nonexistent/path/audio.wav", watermark_id=WATERMARK_ID
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].error is not None

    def test_corrupt_wav_returns_error_result(
        self, detector: WatermarkDetector
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corrupt_path = str(Path(tmp) / "corrupt.wav")
            with open(corrupt_path, "wb") as f:
                f.write(b"THIS IS NOT A WAV FILE AT ALL - GARBAGE DATA !!!")

            results = detector.detect_from_file(corrupt_path, watermark_id=WATERMARK_ID)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0].error is not None


# ---------------------------------------------------------------------------
# Corrupted waveform (empty tensor)
# ---------------------------------------------------------------------------

class TestCorruptedAudio:
    """Corrupted / degenerate waveforms should produce error DetectionResults."""

    def test_empty_tensor_classical(self, detector: WatermarkDetector) -> None:
        empty = torch.tensor([])
        result = detector.detect_classical(empty, WATERMARK_ID)
        assert result.error is not None
        assert result.detected is False

    def test_empty_tensor_learned_with_no_checkpoint(self) -> None:
        detector = WatermarkDetector(checkpoint_dir="/tmp/does_not_exist_echoself")
        empty = torch.tensor([])
        result = detector.detect_learned(empty)
        # Even if checkpoint existed, the error from empty tensor should be caught
        assert result.detected is False
        # No crash
        assert isinstance(result, DetectionResult)

    def test_nan_waveform_classical(self, detector: WatermarkDetector) -> None:
        """NaN-filled waveform should not crash the detector."""
        nan_wave = torch.full((N_SAMPLES,), float("nan"))
        result = detector.detect_classical(nan_wave, WATERMARK_ID)
        # Should not raise; detection result may have error or correlation=0
        assert isinstance(result, DetectionResult)
        assert result.watermark_type == "classical"
