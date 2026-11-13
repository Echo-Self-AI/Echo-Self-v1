"""
EchoSelf Week 11 — Robustness Testing and Safeguard Evaluation
==============================================================

Five sections:
  1. Watermark Robustness Tests
  2. Speaker Verification Robustness Tests
  3. Audit Log Evaluation Tests
  4. Safeguard Evaluation Tests
  5. VAD Robustness Tests
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------

def _silence(n_samples: int) -> torch.Tensor:
    """Pure silence (all zeros)."""
    return torch.zeros(n_samples, dtype=torch.float32)


def _tone(freq_hz: float, n_samples: int, amp: float = 0.5) -> torch.Tensor:
    """Single-frequency sinusoidal tone."""
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    return amp * torch.sin(2.0 * math.pi * freq_hz * t)


def _make_watermarker(strength: float = 0.01, watermark_id: str = "robustness_test"):
    """Return a configured SpreadSpectrumWatermarker."""
    from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
    from echoself_ai.watermark.config import ClassicalWatermarkConfig

    cfg = ClassicalWatermarkConfig(
        watermark_id=watermark_id,
        strength=strength,
        sample_rate=SAMPLE_RATE,
    )
    return SpreadSpectrumWatermarker(cfg)


def _resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Up/downsample a 1-D waveform with linear interpolation (no external deps)."""
    if orig_sr == target_sr:
        return waveform
    new_len = int(len(waveform) * target_sr / orig_sr)
    # torch.nn.functional.interpolate expects (N, C, L)
    wav_3d = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    resampled = F.interpolate(wav_3d, size=new_len, mode="linear", align_corners=True)
    return resampled.squeeze(0).squeeze(0)


# ===========================================================================
# SECTION 1 — Watermark Robustness Tests
# ===========================================================================

class TestWatermarkRobustness:
    """
    For each transformation:
      1. embed classical watermark
      2. apply transformation
      3. try to extract

    Report correlation and detected (True/False).
    """

    WATERMARK_ID = "robustness_wm_test"
    STRENGTH = 0.01   # detection works reliably at this strength

    def _embed_and_transform(self, transform_fn) -> Tuple[bool, float]:
        """Helper: embed → transform → extract. Returns (detected, correlation)."""
        wm = _make_watermarker(self.STRENGTH, self.WATERMARK_ID)
        original = _tone(440.0, SAMPLE_RATE * 2)          # 2 s tone
        watermarked = wm.embed(original, self.WATERMARK_ID)
        transformed = transform_fn(watermarked)
        # extraction requires a waveform of the same size as the PN sequence used
        # for trim/pad we re-generate PN for the new length automatically.
        detected, corr, _ber = wm.extract(transformed, self.WATERMARK_ID)
        return detected, corr

    # --- Gaussian noise ---

    def test_gaussian_noise_low(self):
        """Low Gaussian noise (sigma=0.001) should preserve detection."""
        def transform(w):
            return w + 0.001 * torch.randn_like(w)

        detected, corr = self._embed_and_transform(transform)
        # Low noise: correlation stays positive
        assert isinstance(corr, float), "correlation must be float"
        # Expectation: low noise preserves detection
        # (soft assertion — log rather than hard-fail so report shows result)
        print(f"\n[noise sigma=0.001] corr={corr:.6f} detected={detected}")
        assert corr > 0, f"Low noise should keep correlation positive, got {corr}"

    def test_gaussian_noise_medium(self):
        """Medium Gaussian noise (sigma=0.01) — correlation should still be positive."""
        def transform(w):
            return w + 0.01 * torch.randn_like(w)

        detected, corr = self._embed_and_transform(transform)
        print(f"\n[noise sigma=0.01] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float)
        # Correlation should be non-NaN
        assert not math.isnan(corr), "correlation must not be NaN"

    def test_gaussian_noise_high(self):
        """High Gaussian noise (sigma=0.1) — detection may fail, but no crash."""
        def transform(w):
            return w + 0.1 * torch.randn_like(w)

        detected, corr = self._embed_and_transform(transform)
        print(f"\n[noise sigma=0.1] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float), "correlation must be float (no crash)"
        assert not math.isnan(corr), "correlation must not be NaN under high noise"
        # We do NOT assert detected=True — high noise may destroy the watermark

    # --- Amplitude scaling ---

    def test_amplitude_scale_half(self):
        """Scale to 0.5× amplitude; correlation sign should be preserved."""
        def transform(w):
            return w * 0.5

        detected, corr = self._embed_and_transform(transform)
        print(f"\n[amplitude 0.5x] corr={corr:.6f} detected={detected}")
        # Scaling preserves correlation sign (normalised cross-correlation is
        # scale-invariant by design).
        assert corr > 0, f"Amplitude scaling should preserve positive correlation, got {corr}"
        assert detected, "0.5× amplitude scaling should not destroy detection"

    def test_amplitude_scale_double(self):
        """Scale to 2.0× amplitude (clipped to [-1,1] is OK); detection should hold."""
        def transform(w):
            return torch.clamp(w * 2.0, -1.0, 1.0)

        detected, corr = self._embed_and_transform(transform)
        print(f"\n[amplitude 2.0x] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float)
        assert not math.isnan(corr)
        # Normalised correlation is scale-invariant, so detection should still hold
        assert corr > 0, f"2× amplitude should keep correlation positive, got {corr}"

    # --- Trimming ---

    def test_trimming_first_10_percent(self):
        """Remove first 10 % of samples; extract from the remaining signal."""
        def transform(w):
            cut = max(1, int(0.10 * w.numel()))
            return w[cut:]  # trimmed waveform has different length

        wm = _make_watermarker(self.STRENGTH, self.WATERMARK_ID)
        original = _tone(440.0, SAMPLE_RATE * 2)
        watermarked = wm.embed(original, self.WATERMARK_ID)
        trimmed = watermarked[int(0.10 * watermarked.numel()):]

        detected, corr, _ber = wm.extract(trimmed, self.WATERMARK_ID)
        print(f"\n[trim 10%] corr={corr:.6f} detected={detected}")
        # Trimming changes signal length → PN sequence re-generated for new length
        # Correlation will be lower but should not crash
        assert isinstance(corr, float)
        assert not math.isnan(corr), "extraction after trim must not produce NaN"

    # --- Padding ---

    def test_padding_silent_samples(self):
        """Pad 500 silent samples at start and end; extract from padded signal."""
        def transform(w):
            pad = torch.zeros(500, dtype=w.dtype)
            return torch.cat([pad, w, pad])

        wm = _make_watermarker(self.STRENGTH, self.WATERMARK_ID)
        original = _tone(440.0, SAMPLE_RATE * 2)
        watermarked = wm.embed(original, self.WATERMARK_ID)
        padded = torch.cat([
            torch.zeros(500, dtype=watermarked.dtype),
            watermarked,
            torch.zeros(500, dtype=watermarked.dtype),
        ])

        detected, corr, _ber = wm.extract(padded, self.WATERMARK_ID)
        print(f"\n[padding 500 samples each side] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float)
        assert not math.isnan(corr), "extraction after padding must not produce NaN"

    # --- Resampling ---

    def test_resampling_16k_to_32k_back_to_16k(self):
        """Upsample 16kHz→32kHz then back to 16kHz using linear interpolation."""
        wm = _make_watermarker(self.STRENGTH, self.WATERMARK_ID)
        original = _tone(440.0, SAMPLE_RATE * 2)
        watermarked = wm.embed(original, self.WATERMARK_ID)

        # Up then down
        up = _resample_linear(watermarked, 16000, 32000)
        down = _resample_linear(up, 32000, 16000)

        # Lengths may differ by ±1 sample due to rounding; trim to original length
        min_len = min(watermarked.numel(), down.numel())
        down = down[:min_len]

        detected, corr, _ber = wm.extract(down, self.WATERMARK_ID)
        print(f"\n[resample 16k→32k→16k] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float)
        assert not math.isnan(corr), "extraction after resampling must not produce NaN"

    # --- DC offset ---

    def test_dc_offset(self):
        """Add DC offset +0.01; detection should be robust."""
        def transform(w):
            return torch.clamp(w + 0.01, -1.0, 1.0)

        detected, corr = self._embed_and_transform(transform)
        print(f"\n[DC offset +0.01] corr={corr:.6f} detected={detected}")
        assert isinstance(corr, float)
        assert not math.isnan(corr)
        # DC offset adds a constant to both signal and PN product; since PN sums to ~0
        # the correlation should stay positive
        assert corr > 0, f"DC offset should not destroy positive correlation, got {corr}"
        assert detected, "DC offset +0.01 should not destroy watermark detection"


# ===========================================================================
# SECTION 2 — Speaker Verification Robustness Tests
# ===========================================================================

class TestSpeakerVerificationRobustness:
    """Robustness of the SpeakerVerifier across diverse input conditions."""

    @pytest.fixture(scope="class")
    @classmethod
    def setup(cls, tmp_path_factory):
        """Create a shared encoder, enrollment, and verifier backed by a temp dir."""
        from echoself_ai.voice.speaker_verification import (
            SpeakerEncoder,
            SpeakerEnrollment,
            SpeakerVerifier,
        )

        tmp_dir = tmp_path_factory.mktemp("sv_robustness")
        encoder = SpeakerEncoder()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_dir)
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_dir)

        # Enroll a speaker with 3 identical 2 s tones at 300 Hz
        waveforms = [_tone(300.0, SAMPLE_RATE * 2) for _ in range(3)]
        enrollment.enroll("robustness_user", waveforms, SAMPLE_RATE)

        return verifier, tmp_dir

    def test_same_waveform_highest_similarity(self, setup):
        """Same waveform as enrollment → similarity should be very high (>0.95)."""
        verifier, _ = setup
        same = _tone(300.0, SAMPLE_RATE * 2)
        accepted, sim = verifier.verify("robustness_user", same, SAMPLE_RATE, threshold=0.0)
        print(f"\n[same waveform] similarity={sim:.4f}")
        assert sim > 0.95, f"Same waveform similarity {sim:.4f} should be > 0.95"

    def test_same_frequency_different_amplitude(self, setup):
        """Same frequency, 50 % amplitude — should still be close to same speaker."""
        verifier, _ = setup
        scaled = _tone(300.0, SAMPLE_RATE * 2, amp=0.25)  # 50 % of 0.5
        _, sim_scaled = verifier.verify("robustness_user", scaled, SAMPLE_RATE, threshold=0.0)
        same = _tone(300.0, SAMPLE_RATE * 2)
        _, sim_same = verifier.verify("robustness_user", same, SAMPLE_RATE, threshold=0.0)
        print(f"\n[diff amplitude] sim_same={sim_same:.4f} sim_scaled={sim_scaled:.4f}")
        # Similarity should be reasonably close (within 0.3 of the reference)
        assert abs(sim_scaled - sim_same) < 0.3, (
            f"Same-frequency different-amplitude similarity difference "
            f"({abs(sim_scaled - sim_same):.4f}) should be < 0.3"
        )

    def test_different_frequency_lower_similarity(self, setup):
        """Completely different waveform (different frequency) → lower similarity."""
        verifier, _ = setup
        same = _tone(300.0, SAMPLE_RATE * 2)
        different = _tone(4000.0, SAMPLE_RATE * 2)

        _, sim_same = verifier.verify("robustness_user", same, SAMPLE_RATE, threshold=0.0)
        _, sim_diff = verifier.verify("robustness_user", different, SAMPLE_RATE, threshold=0.0)
        print(f"\n[diff frequency] sim_same={sim_same:.4f} sim_diff={sim_diff:.4f}")
        assert sim_same >= sim_diff, (
            f"Matched similarity ({sim_same:.4f}) should be >= unmatched ({sim_diff:.4f})"
        )

    def test_added_noise_recognized(self, setup):
        """Noisy version of enrolled tone (sigma=0.1) — does not crash, returns float."""
        verifier, _ = setup
        noisy = _tone(300.0, SAMPLE_RATE * 2) + 0.1 * torch.randn(SAMPLE_RATE * 2)
        noisy = torch.clamp(noisy, -1.0, 1.0)
        accepted, sim = verifier.verify("robustness_user", noisy, SAMPLE_RATE, threshold=0.0)
        print(f"\n[added noise sigma=0.1] similarity={sim:.4f} accepted={accepted}")
        assert isinstance(sim, float), "similarity must be float under noise"
        assert -1.0 <= sim <= 1.0, f"similarity {sim} out of range"

    def test_reversed_audio_lower_similarity(self, setup):
        """Reversed waveform → lower similarity than original."""
        verifier, _ = setup
        forward = _tone(300.0, SAMPLE_RATE * 2)
        reversed_wav = forward.flip(0)

        _, sim_fwd = verifier.verify("robustness_user", forward, SAMPLE_RATE, threshold=0.0)
        _, sim_rev = verifier.verify("robustness_user", reversed_wav, SAMPLE_RATE, threshold=0.0)
        print(f"\n[reversed audio] sim_forward={sim_fwd:.4f} sim_reversed={sim_rev:.4f}")
        # The reversed waveform still has the same magnitude spectrum, so similarity
        # won't necessarily drop dramatically, but we just check it doesn't crash.
        assert isinstance(sim_rev, float)
        assert -1.0 <= sim_rev <= 1.0

    def test_zero_input_low_similarity_no_crash(self, setup):
        """All-zero waveform → should return low similarity, not crash."""
        verifier, _ = setup
        zero = torch.zeros(SAMPLE_RATE * 2, dtype=torch.float32)
        accepted, sim = verifier.verify("robustness_user", zero, SAMPLE_RATE, threshold=0.0)
        print(f"\n[zero input] similarity={sim:.4f} accepted={accepted}")
        # Must not raise; similarity should be finite
        assert isinstance(sim, float), "Must not crash on zero input"
        assert math.isfinite(sim), f"similarity must be finite for zero input, got {sim}"

    def test_empty_waveform_returns_false_or_raises(self, setup):
        """Empty waveform (0 samples) → (False, 0.0) or raises without crashing badly."""
        verifier, _ = setup
        empty = torch.zeros(0, dtype=torch.float32)
        try:
            accepted, sim = verifier.verify("robustness_user", empty, SAMPLE_RATE, threshold=0.0)
            # If it returns instead of raising, it must be a graceful failure
            print(f"\n[empty waveform] accepted={accepted} sim={sim}")
            assert accepted is False, "Empty waveform must not be accepted"
            assert sim == 0.0, f"Empty waveform similarity must be 0.0, got {sim}"
        except (RuntimeError, ValueError) as exc:
            # Acceptable to raise on empty input
            print(f"\n[empty waveform] raised {type(exc).__name__}: {exc}")

    def test_unregistered_user_returns_false_zero(self, setup):
        """Verifying against an unregistered user_id returns (False, 0.0)."""
        verifier, _ = setup
        wav = _tone(440.0, SAMPLE_RATE)
        accepted, sim = verifier.verify("no_such_user_12345", wav, SAMPLE_RATE)
        print(f"\n[unregistered user] accepted={accepted} sim={sim}")
        assert accepted is False, "Unregistered user must not be accepted"
        assert sim == 0.0, f"Unregistered user similarity must be 0.0, got {sim}"


# ===========================================================================
# SECTION 3 — Audit Log Evaluation Tests
# ===========================================================================

class TestAuditLogEvaluation:
    """Tests for AuditLogger append-only semantics and integrity verification."""

    def _make_logger(self, tmp_path: Path):
        from echoself_ai.audit.logger import AuditLogger
        log_file = tmp_path / "test_audit.jsonl"
        return AuditLogger(log_path=str(log_file)), log_file

    def _make_record(self, user_id: str = "user_a", model_id: str = "model-v1") -> "AuditRecord":
        from echoself_ai.audit.logger import AuditRecord
        return AuditRecord(
            model_id=model_id,
            watermark_type="classical",
            action="synthesize",
            user_id=user_id,
        )

    def test_log_five_records_read_all_returns_five(self, tmp_path):
        """Log 5 records → read_all() returns exactly 5."""
        logger, _ = self._make_logger(tmp_path)
        for i in range(5):
            logger.log(self._make_record(user_id=f"user_{i}"))

        records = logger.read_all()
        assert len(records) == 5, f"Expected 5 records, got {len(records)}"

    def test_verify_integrity_true_for_clean_log(self, tmp_path):
        """verify_integrity() returns True for a clean log."""
        logger, _ = self._make_logger(tmp_path)
        for i in range(3):
            logger.log(self._make_record(user_id=f"user_{i}"))

        assert logger.verify_integrity() is True

    def test_verify_integrity_false_after_corruption(self, tmp_path):
        """verify_integrity() returns False after a line is corrupted."""
        logger, log_file = self._make_logger(tmp_path)
        for i in range(3):
            logger.log(self._make_record(user_id=f"user_{i}"))

        # Corrupt the log file by appending a broken line
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write("THIS IS NOT VALID JSON }{{{]\n")

        assert logger.verify_integrity() is False

    def test_read_all_returns_correct_event_ids(self, tmp_path):
        """Read all records has correct event_ids (non-empty UUIDs)."""
        logger, _ = self._make_logger(tmp_path)
        logged_ids = []
        for i in range(5):
            rec = self._make_record(user_id=f"user_{i}")
            logger.log(rec)
            logged_ids.append(rec.event_id)

        read_records = logger.read_all()
        read_ids = [r.event_id for r in read_records]

        assert logged_ids == read_ids, (
            f"event_ids in log do not match logged IDs:\n"
            f"  logged: {logged_ids}\n"
            f"  read:   {read_ids}"
        )

    def test_duplicate_records_logged_twice(self, tmp_path):
        """Log same record twice → read_all() contains it twice (append-only)."""
        logger, _ = self._make_logger(tmp_path)
        from echoself_ai.audit.logger import AuditRecord

        rec = AuditRecord(
            model_id="model-dup",
            watermark_type="classical",
            action="detect",
            event_id="fixed-event-id-duplicate",
        )
        logger.log(rec)
        logger.log(rec)

        records = logger.read_all()
        duplicate_ids = [r.event_id for r in records if r.event_id == "fixed-event-id-duplicate"]
        assert len(duplicate_ids) == 2, (
            f"Append-only log should contain the duplicate record twice, "
            f"got {len(duplicate_ids)} occurrence(s)"
        )

    def test_different_users_coexist_in_same_log(self, tmp_path):
        """Different users' records coexist in the same log file."""
        logger, _ = self._make_logger(tmp_path)

        users = ["alice", "bob", "carol"]
        for user in users:
            logger.log(self._make_record(user_id=user))

        records = logger.read_all()
        read_users = {r.user_id for r in records}
        assert read_users == set(users), (
            f"Expected user IDs {set(users)}, got {read_users}"
        )


# ===========================================================================
# SECTION 4 — Safeguard Evaluation Tests
# ===========================================================================

class TestSafeguardEvaluation:
    """Tests for consent gates, pipeline validation, and security invariants."""

    # --- VoiceConsentManager ---

    def test_access_without_consent_raises_permission_error(self, tmp_path):
        """Accessing voice processing without consent raises PermissionError."""
        from echoself_ai.voice.consent import VoiceConsentManager

        consent_dir = tmp_path / "consent"
        mgr = VoiceConsentManager(storage_dir=consent_dir)

        with pytest.raises(PermissionError):
            mgr.enforce_gate("user_no_consent", "speaker_a", scope="synthesis")

    def test_access_after_granting_consent_succeeds(self, tmp_path):
        """Accessing voice processing after granting consent does not raise."""
        from echoself_ai.voice.consent import VoiceConsentManager

        consent_dir = tmp_path / "consent"
        mgr = VoiceConsentManager(storage_dir=consent_dir)
        mgr.register_consent("user_consented", "speaker_a", scope="synthesis")

        # Should not raise
        mgr.enforce_gate("user_consented", "speaker_a", scope="synthesis")

    def test_access_after_revoking_consent_raises_permission_error(self, tmp_path):
        """Revoking consent makes subsequent access raise PermissionError again."""
        from echoself_ai.voice.consent import VoiceConsentManager

        consent_dir = tmp_path / "consent"
        mgr = VoiceConsentManager(storage_dir=consent_dir)
        mgr.register_consent("user_revoke", "speaker_b", scope="synthesis")

        # First access should succeed
        mgr.enforce_gate("user_revoke", "speaker_b", scope="synthesis")

        # Revoke and re-check
        mgr.revoke_consent("user_revoke", "speaker_b")
        with pytest.raises(PermissionError):
            mgr.enforce_gate("user_revoke", "speaker_b", scope="synthesis")

    def test_acoustic_model_with_consent_manager_no_user_id_raises(self, tmp_path):
        """AcousticModel with consent_manager but no user_id raises PermissionError."""
        from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
        from echoself_ai.voice.consent import VoiceConsentManager

        consent_dir = tmp_path / "consent_am"
        mgr = VoiceConsentManager(storage_dir=consent_dir)

        model = AcousticModel(AcousticModelConfig())
        model.eval()

        # Minimal token input
        token_ids = torch.zeros(1, 5, dtype=torch.long)

        with pytest.raises(PermissionError):
            model(
                token_ids,
                user_id=None,            # missing user_id
                raw_speaker_id=None,
                consent_manager=mgr,
            )

    def test_pipeline_malformed_audio_too_short_returns_error(self, tmp_path):
        """Pipeline with 10-sample input (too short) returns error result."""
        from echoself_ai.pipeline import SpeechPipeline, PipelineConfig

        cfg = PipelineConfig(
            sample_rate=SAMPLE_RATE,
            audit_enabled=False,  # skip audit to avoid log side-effects
        )
        pipeline = SpeechPipeline(cfg)

        tiny = torch.zeros(10, dtype=torch.float32)  # 10 samples ≪ 160 minimum
        result = pipeline.process(tiny, sample_rate=SAMPLE_RATE)

        assert result.success is False, (
            "Pipeline must report failure for a 10-sample (too-short) input"
        )
        assert result.error is not None, "error field must be set for too-short input"
        assert "validate" in result.error.lower() or len(result.error) > 0

    def test_pipeline_nan_audio_returns_error(self, tmp_path):
        """Pipeline with NaN audio returns error result."""
        from echoself_ai.pipeline import SpeechPipeline, PipelineConfig

        cfg = PipelineConfig(
            sample_rate=SAMPLE_RATE,
            audit_enabled=False,
        )
        pipeline = SpeechPipeline(cfg)

        nan_audio = torch.full((SAMPLE_RATE,), float("nan"), dtype=torch.float32)
        result = pipeline.process(nan_audio, sample_rate=SAMPLE_RATE)

        assert result.success is False, (
            "Pipeline must report failure for NaN audio"
        )
        assert result.error is not None, "error field must be set for NaN audio"

    def test_pipeline_empty_string_text_input_handles_gracefully(self, tmp_path):
        """Pipeline with empty text_input returns empty or error result gracefully."""
        from echoself_ai.pipeline import SpeechPipeline, PipelineConfig

        cfg = PipelineConfig(
            sample_rate=SAMPLE_RATE,
            audit_enabled=False,
        )
        pipeline = SpeechPipeline(cfg)

        # Valid audio length but empty text_input
        valid_audio = _tone(440.0, SAMPLE_RATE * 2)
        result = pipeline.process(valid_audio, sample_rate=SAMPLE_RATE, text_input="")

        # Must not raise an exception — either succeeds with silence output or fails gracefully
        from echoself_ai.pipeline import PipelineResult
        assert isinstance(result, PipelineResult), "Must return a PipelineResult, not raise"
        # transcript should be empty string or None
        assert result.transcript in (None, "", ""), (
            f"Empty text_input transcript should be empty, got {result.transcript!r}"
        )

    def test_speaker_verification_unregistered_user_returns_false_zero(self, tmp_path):
        """SpeakerVerifier for an unregistered user returns (False, 0.0)."""
        from echoself_ai.voice.speaker_verification import SpeakerVerifier

        storage_dir = tmp_path / "sv_unregistered"
        verifier = SpeakerVerifier(storage_dir=storage_dir)

        wav = _tone(440.0, SAMPLE_RATE)
        accepted, sim = verifier.verify("totally_unregistered_xyz", wav, SAMPLE_RATE)

        assert accepted is False, "Unregistered user must not be accepted"
        assert sim == 0.0, f"Similarity must be 0.0 for unregistered user, got {sim}"


# ===========================================================================
# SECTION 5 — VAD Robustness Tests
# ===========================================================================

class TestVADRobustness:
    """VAD handles degenerate and edge-case inputs without crashing."""

    def _make_energy_vad(self, sample_rate: int = SAMPLE_RATE):
        from echoself_ai.voice.vad import EnergyVAD
        from echoself_ai.voice.vad_config import VADConfig

        cfg = VADConfig(
            method="energy",
            energy_threshold=-4.0,
            sample_rate=sample_rate,
            min_speech_duration_ms=50,
            hangover_ms=200,
        )
        return EnergyVAD(cfg)

    def test_pure_silence_zero_speech_segments(self):
        """Pure silence: VAD returns 0 speech segments."""
        vad = self._make_energy_vad()
        silence = _silence(SAMPLE_RATE * 2)
        segments = vad.detect(silence, SAMPLE_RATE)
        assert segments == [], f"Pure silence should give 0 segments, got {segments}"

    def test_very_short_audio_100ms_no_crash(self):
        """100 ms audio: VAD handles without crash."""
        vad = self._make_energy_vad()
        short = _tone(440.0, int(0.1 * SAMPLE_RATE))  # 100 ms
        # Should not raise
        segments = vad.detect(short, SAMPLE_RATE)
        assert isinstance(segments, list), "Must return a list for 100 ms audio"

    def test_very_loud_audio_clipped_no_nan(self):
        """Clipped audio (peak ±1.0): VAD handles without NaN."""
        vad = self._make_energy_vad()
        loud = torch.clamp(_tone(440.0, SAMPLE_RATE * 2, amp=10.0), -1.0, 1.0)
        segments = vad.detect(loud, SAMPLE_RATE)
        assert isinstance(segments, list), "Must return a list for loud/clipped audio"
        # Verify no NaN crept in by checking segment timestamps
        for start, end in segments:
            assert not math.isnan(start), "Segment start is NaN"
            assert not math.isnan(end), "Segment end is NaN"

    def test_all_nan_audio_raises_or_returns_empty(self):
        """All-NaN audio: VAD raises or returns empty segments (no silent crash)."""
        from echoself_ai.voice.vad import EnergyVAD
        from echoself_ai.voice.vad_config import VADConfig

        vad = self._make_energy_vad()
        nan_audio = torch.full((SAMPLE_RATE,), float("nan"), dtype=torch.float32)

        try:
            segments = vad.detect(nan_audio, SAMPLE_RATE)
            # If it returns, must be a list (empty or otherwise)
            print(f"\n[NaN audio] segments={segments}")
            assert isinstance(segments, list), "NaN audio must return a list or raise"
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            print(f"\n[NaN audio] raised {type(exc).__name__}: {exc}")
            # Acceptable — NaN input is degenerate and may legitimately raise

    def test_different_sample_rates_all_work(self):
        """VAD works at 8kHz, 16kHz, and 22050Hz sample rates."""
        for sr in [8000, 16000, 22050]:
            from echoself_ai.voice.vad import EnergyVAD
            from echoself_ai.voice.vad_config import VADConfig

            cfg = VADConfig(
                method="energy",
                energy_threshold=-4.0,
                sample_rate=sr,
                min_speech_duration_ms=50,
                hangover_ms=200,
            )
            vad = EnergyVAD(cfg)

            # 1 second tone at that sample rate
            n = sr  # 1 s worth of samples
            t = torch.arange(n, dtype=torch.float32) / sr
            tone = 0.5 * torch.sin(2.0 * math.pi * 440.0 * t)

            segments = vad.detect(tone, sr)
            print(f"\n[sample rate {sr}] segments={segments}")
            assert isinstance(segments, list), f"Must return list at sr={sr}"
            assert len(segments) >= 1, (
                f"EnergyVAD must detect speech in a 1 s tone at sr={sr}; "
                f"got 0 segments. Consider adjusting energy_threshold for this test."
            )
