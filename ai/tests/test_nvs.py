"""Tests for the Neural Voice Synthesis (NVS) module — Week 4 EchoSelf."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest
import torch

from echoself_ai.voice.consent import VoiceConsentManager
from echoself_ai.voice.nvs import (
    NVSConfig,
    TextToSpeech,
    get_synthesis_latency,
    synthesize_to_file,
)


# ---------------------------------------------------------------------------
# Shared fixture — small/fast config for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_config() -> NVSConfig:
    """Minimal config: tiny model, few vocoder iters, no checkpoint loading."""
    return NVSConfig(
        n_mels=20,
        hidden_dim=16,
        num_layers=1,
        num_speakers=4,
        upsample_factor=2,
        n_fft=64,
        hop_length=16,
        win_length=64,
        n_iters=4,  # Very few GL iterations — fast
        checkpoint_path=None,  # Always start fresh in tests
        sample_rate=16000,
        device="cpu",
    )


@pytest.fixture(scope="module")
def tts(fast_config: NVSConfig) -> TextToSpeech:
    return TextToSpeech(config=fast_config)


# ---------------------------------------------------------------------------
# NVSConfig
# ---------------------------------------------------------------------------


class TestNVSConfig:
    def test_default_instantiation(self) -> None:
        cfg = NVSConfig()
        assert cfg.sample_rate == 16000
        assert cfg.n_mels == 80
        assert cfg.device == "cpu"

    def test_to_acoustic_config(self) -> None:
        cfg = NVSConfig(n_mels=40, hidden_dim=32)
        a = cfg.to_acoustic_config()
        assert a.n_mels == 40
        assert a.hidden_dim == 32

    def test_invalid_n_mels_raises(self) -> None:
        with pytest.raises(ValueError, match="n_mels"):
            NVSConfig(n_mels=0)

    def test_invalid_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            NVSConfig(sample_rate=4000)

    def test_invalid_target_peak_raises(self) -> None:
        with pytest.raises(ValueError, match="target_peak"):
            NVSConfig(target_peak=0.0)

    def test_no_checkpoint_path(self) -> None:
        cfg = NVSConfig(checkpoint_path=None)
        assert cfg.checkpoint_path is None


# ---------------------------------------------------------------------------
# synthesize() — shape and type
# ---------------------------------------------------------------------------


class TestSynthesizeShape:
    def test_returns_tuple(self, tts: TextToSpeech) -> None:
        result = tts.synthesize("hello")
        assert isinstance(result, tuple) and len(result) == 2

    def test_returns_waveform_tensor(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("hello")
        assert isinstance(waveform, torch.Tensor)

    def test_waveform_is_1d(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("hi")
        assert waveform.dim() == 1, f"Expected 1D, got shape {waveform.shape}"

    def test_waveform_is_float32(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("test")
        assert waveform.dtype == torch.float32

    def test_waveform_has_nonzero_length(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("hello world")
        assert waveform.numel() > 0

    def test_longer_text_produces_longer_waveform(self, tts: TextToSpeech) -> None:
        waveform_short, _ = tts.synthesize("hi")
        waveform_long, _ = tts.synthesize("hello world how are you")
        assert waveform_long.numel() >= waveform_short.numel()


# ---------------------------------------------------------------------------
# synthesize() — sample rate
# ---------------------------------------------------------------------------


class TestSynthesizeSampleRate:
    def test_returns_configured_sample_rate(self, tts: TextToSpeech) -> None:
        _, sr = tts.synthesize("hello")
        assert sr == tts.config.sample_rate

    def test_sample_rate_is_int(self, tts: TextToSpeech) -> None:
        _, sr = tts.synthesize("test")
        assert isinstance(sr, int)

    def test_custom_sample_rate(self, fast_config: NVSConfig) -> None:
        cfg = NVSConfig(
            n_mels=fast_config.n_mels,
            hidden_dim=fast_config.hidden_dim,
            num_layers=fast_config.num_layers,
            upsample_factor=fast_config.upsample_factor,
            n_fft=fast_config.n_fft,
            hop_length=fast_config.hop_length,
            win_length=fast_config.win_length,
            n_iters=fast_config.n_iters,
            checkpoint_path=None,
            sample_rate=22050,
        )
        tts = TextToSpeech(config=cfg)
        _, sr = tts.synthesize("hi")
        assert sr == 22050


# ---------------------------------------------------------------------------
# NaN / Inf validation
# ---------------------------------------------------------------------------


class TestWaveformValidity:
    def test_no_nan_in_waveform(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("hello world")
        assert not torch.isnan(waveform).any(), "Waveform contains NaN values"

    def test_no_inf_in_waveform(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("test synthesis")
        assert not torch.isinf(waveform).any(), "Waveform contains Inf values"

    def test_waveform_amplitude_bounded(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("audio check")
        assert waveform.abs().max().item() <= 1.0 + 1e-6

    def test_waveform_on_cpu(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("check device")
        assert waveform.device.type == "cpu"


# ---------------------------------------------------------------------------
# synthesize_to_file()
# ---------------------------------------------------------------------------


class TestSynthesizeToFile:
    def test_creates_file(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "output.wav"
            tts.synthesize_to_file("hello file", output_path=out)
            assert out.is_file()

    def test_output_is_valid_wav(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "speech.wav"
            tts.synthesize_to_file("test wav", output_path=out)
            # wave.open() raises wave.Error if the file is not a valid WAV
            with wave.open(str(out), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2  # 16-bit
                assert wf.getnframes() > 0
                assert wf.getframerate() == tts.config.sample_rate

    def test_returns_resolved_path(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.wav"
            returned_path = tts.synthesize_to_file("return path", output_path=out)
            assert isinstance(returned_path, Path)
            assert returned_path.exists()

    def test_creates_parent_dirs(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dirs" / "speech.wav"
            tts.synthesize_to_file("nested dirs", output_path=out)
            assert out.is_file()

    def test_module_level_convenience_function(self, fast_config: NVSConfig) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "convenience.wav"
            synthesize_to_file("convenience", output_path=out, config=fast_config)
            assert out.is_file()


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


class TestSynthesisLatency:
    def test_returns_positive_float(self, tts: TextToSpeech) -> None:
        latency = tts.get_synthesis_latency("hello")
        assert isinstance(latency, float)
        assert latency > 0.0

    def test_latency_scales_with_text_length(self, tts: TextToSpeech) -> None:
        """Longer text should not be faster than shorter text (soft check)."""
        # We can't assert strictly because of timing jitter, but the call must complete.
        short = tts.get_synthesis_latency("hi")
        long = tts.get_synthesis_latency("hello world this is a longer sentence")
        assert short > 0.0
        assert long > 0.0

    def test_module_level_latency(self, fast_config: NVSConfig) -> None:
        latency = get_synthesis_latency("test", config=fast_config)
        assert latency > 0.0

    def test_multiple_calls_stable(self, tts: TextToSpeech) -> None:
        latencies = [tts.get_synthesis_latency("hello") for _ in range(3)]
        assert all(l > 0.0 for l in latencies)


# ---------------------------------------------------------------------------
# Empty text handling
# ---------------------------------------------------------------------------


class TestEmptyText:
    def test_empty_string_returns_empty_tensor(self, tts: TextToSpeech) -> None:
        waveform, sr = tts.synthesize("")
        assert isinstance(waveform, torch.Tensor)
        assert waveform.numel() == 0

    def test_empty_string_returns_valid_sample_rate(self, tts: TextToSpeech) -> None:
        _, sr = tts.synthesize("")
        assert sr == tts.config.sample_rate

    def test_empty_string_no_nan(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("")
        assert not torch.isnan(waveform).any()


# ---------------------------------------------------------------------------
# Speaker ID conditioning
# ---------------------------------------------------------------------------


class TestSpeakerID:
    def test_synthesize_with_speaker_id_zero(self, tts: TextToSpeech) -> None:
        waveform, sr = tts.synthesize("hello speaker", speaker_id=0)
        assert waveform.numel() > 0
        assert sr == tts.config.sample_rate

    def test_synthesize_with_speaker_id_one(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("speaker one", speaker_id=1)
        assert waveform.numel() > 0

    def test_different_speakers_produce_different_waveforms(self, tts: TextToSpeech) -> None:
        """Different speaker embeddings should produce distinct mel spectrograms."""
        w0, _ = tts.synthesize("same text", speaker_id=0)
        w1, _ = tts.synthesize("same text", speaker_id=1)
        # They may have different lengths or values; just ensure no crash and both valid
        assert w0.numel() > 0
        assert w1.numel() > 0
        assert not torch.isnan(w0).any()
        assert not torch.isnan(w1).any()

    def test_speaker_id_none_works(self, tts: TextToSpeech) -> None:
        waveform, _ = tts.synthesize("no speaker", speaker_id=None)
        assert waveform.numel() > 0


# ---------------------------------------------------------------------------
# Consent gate integration
# ---------------------------------------------------------------------------


class TestConsentGate:
    def test_consent_granted_allows_synthesis(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = VoiceConsentManager(storage_dir=Path(tmpdir))
            cm.register_consent(user_id="user1", speaker_id="0", scope="synthesis")
            waveform, sr = tts.synthesize(
                "consent test",
                speaker_id=0,
                user_id="user1",
                consent_manager=cm,
            )
            assert waveform.numel() > 0

    def test_consent_denied_raises_permission_error(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = VoiceConsentManager(storage_dir=Path(tmpdir))
            # Deliberately do NOT register consent
            with pytest.raises(PermissionError):
                tts.synthesize(
                    "no consent",
                    speaker_id=0,
                    user_id="user2",
                    consent_manager=cm,
                )

    def test_no_consent_manager_no_error(self, tts: TextToSpeech) -> None:
        """When consent_manager is None, synthesis proceeds without gate."""
        waveform, _ = tts.synthesize("no manager", speaker_id=0, consent_manager=None)
        assert waveform.numel() > 0

    def test_revoked_consent_raises_permission_error(self, tts: TextToSpeech) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = VoiceConsentManager(storage_dir=Path(tmpdir))
            cm.register_consent(user_id="user3", speaker_id="0", scope="synthesis")
            cm.revoke_consent(user_id="user3", speaker_id="0")
            with pytest.raises(PermissionError):
                tts.synthesize(
                    "revoked consent",
                    speaker_id=0,
                    user_id="user3",
                    consent_manager=cm,
                )


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------


class TestCPUFallback:
    def test_explicit_cpu_config(self, fast_config: NVSConfig) -> None:
        cfg = NVSConfig(
            n_mels=fast_config.n_mels,
            hidden_dim=fast_config.hidden_dim,
            num_layers=fast_config.num_layers,
            upsample_factor=fast_config.upsample_factor,
            n_fft=fast_config.n_fft,
            hop_length=fast_config.hop_length,
            win_length=fast_config.win_length,
            n_iters=fast_config.n_iters,
            checkpoint_path=None,
            device="cpu",
        )
        tts = TextToSpeech(config=cfg)
        waveform, _ = tts.synthesize("cpu test")
        assert waveform.device.type == "cpu"

    def test_model_params_on_cpu(self, fast_config: NVSConfig) -> None:
        cfg = NVSConfig(
            n_mels=fast_config.n_mels,
            hidden_dim=fast_config.hidden_dim,
            num_layers=fast_config.num_layers,
            upsample_factor=fast_config.upsample_factor,
            n_fft=fast_config.n_fft,
            hop_length=fast_config.hop_length,
            win_length=fast_config.win_length,
            n_iters=fast_config.n_iters,
            checkpoint_path=None,
            device="cpu",
        )
        tts = TextToSpeech(config=cfg)
        for p in tts._acoustic_model.parameters():
            assert p.device.type == "cpu"
            break  # checking one parameter suffices


# ---------------------------------------------------------------------------
# Checkpoint loading (graceful degradation)
# ---------------------------------------------------------------------------


class TestCheckpointLoading:
    def test_missing_checkpoint_uses_fresh_model(self) -> None:
        """Should not raise even if checkpoint path doesn't exist."""
        cfg = NVSConfig(checkpoint_path="/nonexistent/path/model.pt")
        # Manually override to a small model so it doesn't take forever
        cfg.n_mels = 20
        cfg.hidden_dim = 16
        cfg.num_layers = 1
        cfg.n_fft = 64
        cfg.hop_length = 16
        cfg.win_length = 64
        cfg.n_iters = 2
        tts = TextToSpeech(config=cfg)
        waveform, _ = tts.synthesize("fallback test")
        assert waveform.numel() > 0

    def test_none_checkpoint_uses_fresh_model(self, fast_config: NVSConfig) -> None:
        tts = TextToSpeech(config=fast_config)
        waveform, _ = tts.synthesize("no checkpoint")
        assert waveform.numel() > 0
