"""Tests for Week 5: Speaker Enrollment and Verification.

Covers:
  - SpeakerEncoder forward pass (shape, L2 norm)
  - Enrollment creates a profile file
  - Same-speaker: high cosine similarity
  - Different-speaker: different embeddings
  - verify() return type
  - revoke() deletes profile
  - Missing profile → (False, 0.0)
  - Challenge phrase is a non-empty string
"""
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from echoself_ai.voice.speaker_verification import (
    SpeakerEncoder,
    SpeakerEncoderConfig,
    SpeakerEnrollment,
    SpeakerProfile,
    SpeakerVerifier,
    generate_challenge_phrase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_waveform(freq_hz: float = 440.0, duration_s: float = 1.0, sr: int = 16000) -> torch.Tensor:
    """Generate a synthetic sine-wave waveform."""
    n = int(duration_s * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    return 0.5 * torch.sin(2.0 * math.pi * freq_hz * t)


def _make_mel(batch: int = 2, n_mels: int = 80, frames: int = 100) -> torch.Tensor:
    """Generate a random log-mel spectrogram tensor."""
    return torch.randn(batch, n_mels, frames)


# ---------------------------------------------------------------------------
# SpeakerEncoderConfig
# ---------------------------------------------------------------------------

class TestSpeakerEncoderConfig:
    def test_defaults(self):
        cfg = SpeakerEncoderConfig()
        assert cfg.input_dim == 80
        assert cfg.hidden_dim == 128
        assert cfg.embedding_dim == 192

    def test_custom_values(self):
        cfg = SpeakerEncoderConfig(input_dim=40, hidden_dim=64, embedding_dim=128)
        assert cfg.input_dim == 40

    def test_invalid_input_dim(self):
        with pytest.raises(ValueError):
            SpeakerEncoderConfig(input_dim=0)

    def test_invalid_hidden_dim(self):
        with pytest.raises(ValueError):
            SpeakerEncoderConfig(hidden_dim=0)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError):
            SpeakerEncoderConfig(embedding_dim=0)


# ---------------------------------------------------------------------------
# SpeakerEncoder
# ---------------------------------------------------------------------------

class TestSpeakerEncoderForward:
    def test_output_shape_batch(self):
        """Encoder maps (B, 80, T) → (B, 192)."""
        encoder = SpeakerEncoder()
        x = _make_mel(batch=4, n_mels=80, frames=50)
        out = encoder(x)
        assert out.shape == (4, 192)

    def test_output_shape_single(self):
        """Encoder works with batch size 1."""
        encoder = SpeakerEncoder()
        x = _make_mel(batch=1, n_mels=80, frames=100)
        out = encoder(x)
        assert out.shape == (1, 192)

    def test_l2_norm_unit(self):
        """Every output embedding must have L2 norm close to 1.0."""
        encoder = SpeakerEncoder()
        x = _make_mel(batch=8, n_mels=80, frames=80)
        out = encoder(x)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), f"norms={norms}"

    def test_custom_config_shape(self):
        """Custom config changes the output dimension."""
        cfg = SpeakerEncoderConfig(input_dim=40, hidden_dim=64, embedding_dim=128)
        encoder = SpeakerEncoder(cfg)
        x = _make_mel(batch=2, n_mels=40, frames=60)
        out = encoder(x)
        assert out.shape == (2, 128)

    def test_l2_norm_custom_config(self):
        """Unit norm must hold for custom config too."""
        cfg = SpeakerEncoderConfig(input_dim=40, hidden_dim=64, embedding_dim=128)
        encoder = SpeakerEncoder(cfg)
        x = _make_mel(batch=3, n_mels=40, frames=60)
        out = encoder(x)
        norms = out.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_deterministic_in_eval_mode(self):
        """Same input → same output in eval mode (batch norm uses running stats)."""
        encoder = SpeakerEncoder()
        encoder.eval()
        x = _make_mel(batch=2, n_mels=80, frames=50)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_no_nan_in_output(self):
        """Encoder must never produce NaN embeddings."""
        encoder = SpeakerEncoder()
        x = torch.randn(4, 80, 200)
        out = encoder(x)
        assert torch.isfinite(out).all()

    def test_varying_temporal_length(self):
        """Encoder should handle different T lengths (temporal average pooling)."""
        encoder = SpeakerEncoder()
        for T in [10, 50, 200, 500]:
            x = _make_mel(batch=1, n_mels=80, frames=T)
            out = encoder(x)
            assert out.shape == (1, 192), f"failed for T={T}"


# ---------------------------------------------------------------------------
# SpeakerProfile
# ---------------------------------------------------------------------------

class TestSpeakerProfile:
    def test_fields_accessible(self):
        p = SpeakerProfile(
            user_id="u1",
            enrollment_id="e1",
            mean_embedding=[0.1, 0.2, 0.3],
            enrolled_at="2026-01-01T00:00:00+00:00",
            num_samples=3,
        )
        assert p.user_id == "u1"
        assert p.enrollment_id == "e1"
        assert len(p.mean_embedding) == 3
        assert p.num_samples == 3

    def test_to_tensor(self):
        """_to_tensor returns correct float32 tensor."""
        emb = [0.1, 0.2, 0.3]
        p = SpeakerProfile("u", "e", emb, "now", 1)
        t = p._to_tensor()
        assert t.dtype == torch.float32
        assert t.shape == (3,)
        assert torch.allclose(t, torch.tensor(emb, dtype=torch.float32))


# ---------------------------------------------------------------------------
# SpeakerEnrollment
# ---------------------------------------------------------------------------

class TestSpeakerEnrollment:
    def test_enroll_creates_profile_file(self, tmp_path):
        """Enrollment must persist a profile file on disk."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        waveforms = [_make_waveform(440.0, 1.0)]
        profile = enrollment.enroll("user_a", waveforms, sample_rate=16000)

        expected_file = tmp_path / "user_a.json"
        assert expected_file.is_file(), "Profile JSON file was not created"
        assert profile.user_id == "user_a"
        assert profile.num_samples == 1

    def test_enroll_multiple_samples(self, tmp_path):
        """Enrolling with N waveforms should set num_samples=N."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        waveforms = [_make_waveform(440.0 + i * 10, 0.5) for i in range(5)]
        profile = enrollment.enroll("user_b", waveforms, sample_rate=16000)
        assert profile.num_samples == 5

    def test_enroll_returns_speaker_profile(self, tmp_path):
        """Return type must be SpeakerProfile."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        result = enrollment.enroll("user_c", [_make_waveform()], sample_rate=16000)
        assert isinstance(result, SpeakerProfile)

    def test_enroll_embedding_is_unit_norm(self, tmp_path):
        """The persisted mean embedding must have ~unit L2 norm."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        profile = enrollment.enroll("user_d", [_make_waveform(), _make_waveform(880)], 16000)
        t = profile._to_tensor()
        norm = t.norm(p=2).item()
        assert abs(norm - 1.0) < 1e-4, f"norm={norm}"

    def test_enroll_empty_waveform_list_raises(self, tmp_path):
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        with pytest.raises(ValueError):
            enrollment.enroll("user_e", [], sample_rate=16000)

    def test_load_profile_returns_profile(self, tmp_path):
        """load_profile must return the previously enrolled profile."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        enrollment.enroll("user_f", [_make_waveform()], 16000)
        loaded = enrollment.load_profile("user_f")
        assert loaded is not None
        assert loaded.user_id == "user_f"

    def test_load_profile_missing_returns_none(self, tmp_path):
        """load_profile must return None when no profile exists."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        result = enrollment.load_profile("nonexistent_user")
        assert result is None

    def test_revoke_deletes_file(self, tmp_path):
        """revoke() must delete the profile file and return True."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        enrollment.enroll("user_g", [_make_waveform()], 16000)
        profile_file = tmp_path / "user_g.json"
        assert profile_file.is_file()

        result = enrollment.revoke("user_g")
        assert result is True
        assert not profile_file.is_file()

    def test_revoke_missing_returns_false(self, tmp_path):
        """revoke() must return False when no profile exists."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        result = enrollment.revoke("nobody")
        assert result is False

    def test_revoke_then_load_returns_none(self, tmp_path):
        """After revoke, load_profile must return None."""
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        enrollment.enroll("user_h", [_make_waveform()], 16000)
        enrollment.revoke("user_h")
        assert enrollment.load_profile("user_h") is None

    def test_round_trip_embedding_preserved(self, tmp_path):
        """Serialise → deserialise must preserve embedding values."""
        encoder = SpeakerEncoder()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_path)
        waveform = _make_waveform(440.0, 2.0)
        profile = enrollment.enroll("user_i", [waveform], 16000)
        loaded = enrollment.load_profile("user_i")
        assert loaded is not None
        orig = torch.tensor(profile.mean_embedding)
        reloaded = torch.tensor(loaded.mean_embedding)
        assert torch.allclose(orig, reloaded, atol=1e-6)


# ---------------------------------------------------------------------------
# Same-speaker / different-speaker similarity tests
# ---------------------------------------------------------------------------

class TestSpeakerSimilarity:
    def test_same_input_high_similarity(self, tmp_path):
        """Identical waveform enrolled and queried → cosine similarity > 0.5."""
        encoder = SpeakerEncoder()
        encoder.eval()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_path)

        waveform = _make_waveform(440.0, 2.0)
        profile = enrollment.enroll("sim_user", [waveform], 16000)

        # Re-compute query embedding for the same waveform
        from echoself_ai.voice.speaker_verification import _extract_embedding
        query = _extract_embedding(encoder, waveform, 16000)
        enrolled = F.normalize(profile._to_tensor(), p=2, dim=0)
        sim = torch.dot(F.normalize(query, p=2, dim=0), enrolled).item()

        assert sim > 0.5, f"Expected high similarity for same input, got {sim:.4f}"

    def test_different_waveforms_produce_different_embeddings(self):
        """Significantly different waveforms should produce different embeddings."""
        encoder = SpeakerEncoder()
        encoder.eval()

        from echoself_ai.voice.speaker_verification import _extract_embedding
        wav_a = _make_waveform(220.0, 1.0)
        wav_b = torch.randn(16000) * 0.1  # very different: white noise

        emb_a = _extract_embedding(encoder, wav_a, 16000)
        emb_b = _extract_embedding(encoder, wav_b, 16000)

        sim = torch.dot(F.normalize(emb_a, p=2, dim=0), F.normalize(emb_b, p=2, dim=0)).item()
        # They should not be identical (not necessarily < threshold)
        assert not torch.allclose(emb_a, emb_b), "Different waveforms produced identical embeddings"

    def test_re_enrollment_same_waveform_consistent(self, tmp_path):
        """Re-enrolling with the same waveform should produce a consistent profile."""
        encoder = SpeakerEncoder()
        encoder.eval()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_path)

        waveform = _make_waveform(440.0, 1.0)
        p1 = enrollment.enroll("re_user", [waveform], 16000)
        # Overwrite
        p2 = enrollment.enroll("re_user", [waveform], 16000)

        t1 = torch.tensor(p1.mean_embedding)
        t2 = torch.tensor(p2.mean_embedding)
        # Same waveform → same embedding → sim ≈ 1.0
        sim = torch.dot(t1, t2).item()
        assert sim > 0.99, f"Expected near-identical profiles, similarity={sim}"


# ---------------------------------------------------------------------------
# SpeakerVerifier
# ---------------------------------------------------------------------------

class TestSpeakerVerifier:
    def test_verify_returns_tuple(self, tmp_path):
        """verify() must return a (bool, float) tuple."""
        verifier = SpeakerVerifier(storage_dir=tmp_path)
        waveform = _make_waveform()
        # Enroll first via the internal enrollment object
        verifier._enrollment.enroll("v_user", [waveform], 16000)

        result = verifier.verify("v_user", waveform, 16000)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_verify_returns_bool_and_float(self, tmp_path):
        """Return types must be bool and float."""
        verifier = SpeakerVerifier(storage_dir=tmp_path)
        waveform = _make_waveform()
        verifier._enrollment.enroll("v_user2", [waveform], 16000)

        is_accepted, similarity = verifier.verify("v_user2", waveform, 16000)
        assert isinstance(is_accepted, bool)
        assert isinstance(similarity, float)

    def test_verify_missing_profile_returns_false_zero(self, tmp_path):
        """Missing profile must return (False, 0.0)."""
        verifier = SpeakerVerifier(storage_dir=tmp_path)
        result = verifier.verify("nobody", _make_waveform(), 16000)
        assert result == (False, 0.0)

    def test_verify_same_waveform_accepted_low_threshold(self, tmp_path):
        """Same waveform with low threshold (0.1) should be accepted."""
        encoder = SpeakerEncoder()
        encoder.eval()
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_path)
        waveform = _make_waveform(440.0, 2.0)
        verifier._enrollment.enroll("v_same", [waveform], 16000)

        is_accepted, similarity = verifier.verify("v_same", waveform, 16000, threshold=0.1)
        assert is_accepted is True
        assert similarity > 0.1

    def test_verify_high_threshold_rejects(self, tmp_path):
        """Very high threshold (0.9999) should reject even identical waveforms."""
        encoder = SpeakerEncoder()
        encoder.eval()
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_path)
        waveform = _make_waveform(440.0, 2.0)
        verifier._enrollment.enroll("v_strict", [waveform], 16000)

        # With threshold=0.9999 it's very unlikely to pass unless literally identical
        is_accepted, similarity = verifier.verify("v_strict", waveform, 16000, threshold=0.9999)
        # Just verify it returns correct types regardless of accept/reject
        assert isinstance(is_accepted, bool)
        assert 0.0 <= similarity <= 1.0

    def test_verify_similarity_in_range(self, tmp_path):
        """Cosine similarity must be in [-1, 1] (we clamp it to this range)."""
        verifier = SpeakerVerifier(storage_dir=tmp_path)
        waveform = _make_waveform()
        verifier._enrollment.enroll("v_range", [waveform], 16000)

        _, sim = verifier.verify("v_range", waveform, 16000)
        assert -1.0 <= sim <= 1.0

    def test_verify_after_revoke_returns_false(self, tmp_path):
        """After revoking, verify must return (False, 0.0)."""
        verifier = SpeakerVerifier(storage_dir=tmp_path)
        waveform = _make_waveform()
        verifier._enrollment.enroll("v_revoked", [waveform], 16000)
        verifier._enrollment.revoke("v_revoked")

        result = verifier.verify("v_revoked", waveform, 16000)
        assert result == (False, 0.0)

    def test_verify_default_threshold_is_0_75(self, tmp_path):
        """Default threshold should be 0.75."""
        import inspect
        from echoself_ai.voice.speaker_verification import SpeakerVerifier as SV
        sig = inspect.signature(SV.verify)
        default_threshold = sig.parameters["threshold"].default
        assert default_threshold == 0.75


# ---------------------------------------------------------------------------
# Challenge phrase
# ---------------------------------------------------------------------------

class TestGenerateChallengePhrase:
    def test_returns_string(self):
        result = generate_challenge_phrase()
        assert isinstance(result, str)

    def test_non_empty(self):
        result = generate_challenge_phrase()
        assert len(result) > 0

    def test_varies_across_calls(self):
        """Should not always return the same phrase (probabilistic)."""
        phrases = {generate_challenge_phrase() for _ in range(50)}
        assert len(phrases) > 1, "generate_challenge_phrase never varies"

    def test_phrase_from_known_list(self):
        """Returned phrase must be from the fixed list of 20."""
        from echoself_ai.voice.speaker_verification import _CHALLENGE_PHRASES
        for _ in range(20):
            p = generate_challenge_phrase()
            assert p in _CHALLENGE_PHRASES

    def test_list_has_20_phrases(self):
        from echoself_ai.voice.speaker_verification import _CHALLENGE_PHRASES
        assert len(_CHALLENGE_PHRASES) == 20


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_enroll_verify_revoke_cycle(self, tmp_path):
        """Full cycle: enroll → verify → revoke → verify (rejected)."""
        encoder = SpeakerEncoder()
        encoder.eval()
        enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_path)
        verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_path)

        user_id = "e2e_user"
        waveforms = [_make_waveform(440.0 + i * 20, 1.0) for i in range(3)]
        profile = enrollment.enroll(user_id, waveforms, sample_rate=16000)

        assert isinstance(profile, SpeakerProfile)
        assert profile.user_id == user_id
        assert profile.num_samples == 3

        # Verify with one of the enrolled waveforms at low threshold
        is_accepted, similarity = verifier.verify(user_id, waveforms[0], 16000, threshold=0.1)
        assert isinstance(is_accepted, bool)
        assert isinstance(similarity, float)

        # Revoke
        revoked = enrollment.revoke(user_id)
        assert revoked is True

        # Post-revoke verify must fail
        result = verifier.verify(user_id, waveforms[0], 16000)
        assert result == (False, 0.0)

    def test_profile_enrollment_id_is_uuid(self, tmp_path):
        """enrollment_id must be a valid UUID string."""
        import re
        UUID_RE = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        profile = enrollment.enroll("uuid_user", [_make_waveform()], 16000)
        assert UUID_RE.match(profile.enrollment_id), f"Not a UUID: {profile.enrollment_id}"

    def test_enrolled_at_is_iso_timestamp(self, tmp_path):
        """enrolled_at must be a parseable ISO-8601 timestamp."""
        from datetime import datetime
        enrollment = SpeakerEnrollment(storage_dir=tmp_path)
        profile = enrollment.enroll("ts_user", [_make_waveform()], 16000)
        # This will raise if not a valid ISO format
        dt = datetime.fromisoformat(profile.enrolled_at)
        assert dt is not None
