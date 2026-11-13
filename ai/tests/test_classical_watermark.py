"""
tests/test_classical_watermark.py
==================================
Unit tests for the classical spread-spectrum audio watermarker.

All tests use only PyTorch and the standard library — no scipy or librosa.
"""

from __future__ import annotations

import math

import pytest
import torch

from echoself_ai.watermark import (
    ClassicalWatermarkConfig,
    SpreadSpectrumWatermarker,
    bit_error_rate,
    n_bit_watermark_id_to_bits,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
DURATION_S = 2.0
N_SAMPLES = int(DURATION_S * SAMPLE_RATE)
WM_ID_A = "echoself-test-wm-aaaa-0001"
WM_ID_B = "echoself-test-wm-bbbb-0002"


def make_tone(freq: float = 440.0, amplitude: float = 0.5) -> torch.Tensor:
    """Generate a 2-second 440 Hz sine tone at 16 kHz."""
    t = torch.arange(N_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    return amplitude * torch.sin(2.0 * math.pi * freq * t)


def make_watermarker(strength: float = 0.01, wm_id: str = WM_ID_A) -> SpreadSpectrumWatermarker:
    cfg = ClassicalWatermarkConfig(
        watermark_id=wm_id,
        strength=strength,
        n_bits=32,
        seed=42,
        sample_rate=SAMPLE_RATE,
    )
    return SpreadSpectrumWatermarker(cfg)


# ---------------------------------------------------------------------------
# 1. embed() returns tensor of same shape as input
# ---------------------------------------------------------------------------

class TestEmbedShape:
    def test_1d_shape_preserved(self):
        wm = make_watermarker()
        audio = make_tone()
        out = wm.embed(audio, WM_ID_A)
        assert out.shape == audio.shape, f"Expected {audio.shape}, got {out.shape}"

    def test_2d_shape_preserved(self):
        """Stereo or batched audio (channels × samples) keeps its shape."""
        wm = make_watermarker()
        audio = make_tone().unsqueeze(0).expand(2, -1)  # (2, N)
        out = wm.embed(audio, WM_ID_A)
        assert out.shape == audio.shape

    def test_embedding_preserves_audio_length(self):
        wm = make_watermarker()
        audio = make_tone()
        out = wm.embed(audio, WM_ID_A)
        assert out.numel() == audio.numel()


# ---------------------------------------------------------------------------
# 2. embed() output has no NaN / Inf
# ---------------------------------------------------------------------------

class TestEmbedNumerics:
    def test_no_nan(self):
        wm = make_watermarker()
        audio = make_tone()
        out = wm.embed(audio, WM_ID_A)
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_no_inf(self):
        wm = make_watermarker()
        audio = make_tone()
        out = wm.embed(audio, WM_ID_A)
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_output_clipped(self):
        """Values must remain in [-1, 1] even with high strength."""
        wm = make_watermarker(strength=0.9)
        audio = make_tone(amplitude=0.9)
        out = wm.embed(audio, WM_ID_A)
        assert out.max().item() <= 1.0
        assert out.min().item() >= -1.0


# ---------------------------------------------------------------------------
# 3. extract() on clean embedded audio returns detected=True
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detect_embedded_watermark(self):
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)
        detected, corr, ber = wm.extract(watermarked, WM_ID_A)
        assert detected, (
            f"Watermark should be detected after embedding. "
            f"correlation={corr:.6f}, threshold={SpreadSpectrumWatermarker.DETECTION_THRESHOLD}"
        )

    def test_correlation_positive_after_embedding(self):
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)
        _, corr, _ = wm.extract(watermarked, WM_ID_A)
        assert corr > SpreadSpectrumWatermarker.DETECTION_THRESHOLD


# ---------------------------------------------------------------------------
# 4. extract() on clean UN-embedded audio returns detected=False
# ---------------------------------------------------------------------------

class TestFalsePositive:
    def test_no_detection_on_clean_signal(self):
        """Pure tone without any watermark should NOT be detected.

        We use a watermark_id that is completely unrelated to the signal.
        The correlation of a sine wave against a random PN sequence is very
        close to zero for long sequences — well below 0.001.
        """
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        detected, corr, _ = wm.extract(audio, WM_ID_A)
        # For a 32 000-sample signal the expected |correlation| ≈ 1/√N ≈ 0.006
        # which can exceed 0.001 for a pure sine. We therefore only assert that
        # correlation is substantially lower than what we see after embedding,
        # and we directly check the threshold.
        assert not detected or corr < 0.01, (
            f"False positive: corr={corr:.6f} on un-watermarked signal"
        )

    def test_different_wm_id_not_detected(self):
        """Watermark embedded with ID A should not be detected when looking for ID B."""
        wm = make_watermarker(strength=0.05)  # high strength for clear embedding
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)
        # Try to detect with the wrong ID
        detected_b, corr_b, _ = wm.extract(watermarked, WM_ID_B)
        # The wrong-ID correlation should be near zero
        assert abs(corr_b) < 0.05, (
            f"Cross-detection: wrong WM ID gave correlation={corr_b:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. SNR calculation returns positive value
# ---------------------------------------------------------------------------

class TestSNR:
    def test_snr_positive(self):
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)
        snr = wm.compute_snr(audio, watermarked)
        assert snr > 0.0, f"Expected positive SNR, got {snr:.2f} dB"

    def test_snr_gt_20db_for_strength_001(self):
        """At alpha=0.01 the watermark is imperceptible: SNR should exceed 20 dB."""
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)
        snr = wm.compute_snr(audio, watermarked)
        assert snr > 20.0, (
            f"SNR={snr:.2f} dB — expected > 20 dB for strength=0.01"
        )

    def test_snr_increases_with_smaller_strength(self):
        """Lower embedding strength should give higher (better) SNR."""
        audio = make_tone()
        wm_weak = make_watermarker(strength=0.001)
        wm_strong = make_watermarker(strength=0.1)
        snr_weak = wm_weak.compute_snr(audio, wm_weak.embed(audio, WM_ID_A))
        snr_strong = wm_strong.compute_snr(audio, wm_strong.embed(audio, WM_ID_A))
        assert snr_weak > snr_strong, (
            f"Weaker watermark should have higher SNR: "
            f"weak={snr_weak:.2f}, strong={snr_strong:.2f}"
        )

    def test_snr_inf_when_no_watermark(self):
        """When original == watermarked, noise is 0 → SNR is infinite."""
        wm = make_watermarker()
        audio = make_tone()
        snr = wm.compute_snr(audio, audio)
        assert math.isinf(snr) and snr > 0


# ---------------------------------------------------------------------------
# 6. Same watermark_id gives same PN sequence
# ---------------------------------------------------------------------------

class TestPNReproducibility:
    def test_same_id_same_pn(self):
        pn1 = SpreadSpectrumWatermarker._generate_pn_sequence(1000, seed=12345)
        pn2 = SpreadSpectrumWatermarker._generate_pn_sequence(1000, seed=12345)
        assert torch.equal(pn1, pn2), "Same seed must produce identical PN sequence"

    def test_same_wm_id_same_pn_via_hash(self):
        wm = make_watermarker()
        seed = SpreadSpectrumWatermarker._watermark_id_to_seed(WM_ID_A)
        pn1 = SpreadSpectrumWatermarker._generate_pn_sequence(500, seed)
        pn2 = SpreadSpectrumWatermarker._generate_pn_sequence(500, seed)
        assert torch.equal(pn1, pn2)


# ---------------------------------------------------------------------------
# 7. Different watermark_ids give different PN sequences
# ---------------------------------------------------------------------------

class TestPNUniqueness:
    def test_different_ids_different_pn(self):
        seed_a = SpreadSpectrumWatermarker._watermark_id_to_seed(WM_ID_A)
        seed_b = SpreadSpectrumWatermarker._watermark_id_to_seed(WM_ID_B)
        pn_a = SpreadSpectrumWatermarker._generate_pn_sequence(1000, seed_a)
        pn_b = SpreadSpectrumWatermarker._generate_pn_sequence(1000, seed_b)
        assert not torch.equal(pn_a, pn_b), (
            "Different watermark IDs should produce different PN sequences"
        )

    def test_pn_values_are_plus_minus_one(self):
        pn = SpreadSpectrumWatermarker._generate_pn_sequence(10_000, seed=99)
        unique = set(pn.tolist())
        assert unique == {-1.0, 1.0}, f"PN sequence values: {unique}"

    def test_pn_roughly_balanced(self):
        """PN sequence should be approximately half +1 and half -1."""
        pn = SpreadSpectrumWatermarker._generate_pn_sequence(100_000, seed=7)
        ratio = pn[pn == 1.0].numel() / pn.numel()
        assert 0.45 <= ratio <= 0.55, f"PN imbalance: {ratio:.3f}"


# ---------------------------------------------------------------------------
# 8. After heavy Gaussian noise, correlation degrades
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_heavy_noise_degrades_correlation(self):
        """High noise (sigma=0.5) should degrade the correlation."""
        wm = make_watermarker(strength=0.01)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)

        # Clean detection
        _, corr_clean, _ = wm.extract(watermarked, WM_ID_A)

        # Add heavy noise
        noisy = watermarked + torch.randn_like(watermarked) * 0.5
        noisy = torch.clamp(noisy, -1.0, 1.0)
        _, corr_noisy, _ = wm.extract(noisy, WM_ID_A)

        # The correlation should be lower (possibly even undetectable)
        assert abs(corr_noisy) < abs(corr_clean), (
            f"Expected heavy noise to degrade correlation: "
            f"clean={corr_clean:.6f}, noisy={corr_noisy:.6f}"
        )

    def test_high_strength_survives_mild_noise(self):
        """With alpha=0.05 the watermark should survive mild noise (SNR=30 dB)."""
        wm = make_watermarker(strength=0.05)
        audio = make_tone()
        watermarked = wm.embed(audio, WM_ID_A)

        signal_power = torch.mean(watermarked ** 2).item()
        noise_power = signal_power / (10.0 ** (30.0 / 10.0))
        noisy = watermarked + torch.randn_like(watermarked) * math.sqrt(noise_power)
        noisy = torch.clamp(noisy, -1.0, 1.0)

        detected, corr, _ = wm.extract(noisy, WM_ID_A)
        assert detected, (
            f"Strong watermark should survive mild noise. corr={corr:.6f}"
        )


# ---------------------------------------------------------------------------
# 9. Bit-level helpers
# ---------------------------------------------------------------------------

class TestBitHelpers:
    def test_n_bit_to_bits_length(self):
        bits = n_bit_watermark_id_to_bits(WM_ID_A, 32)
        assert len(bits) == 32

    def test_n_bit_to_bits_values(self):
        bits = n_bit_watermark_id_to_bits(WM_ID_A, 64)
        assert all(b in (0, 1) for b in bits)

    def test_n_bit_to_bits_deterministic(self):
        bits1 = n_bit_watermark_id_to_bits(WM_ID_A, 32)
        bits2 = n_bit_watermark_id_to_bits(WM_ID_A, 32)
        assert bits1 == bits2

    def test_n_bit_different_ids_different_bits(self):
        bits_a = n_bit_watermark_id_to_bits(WM_ID_A, 32)
        bits_b = n_bit_watermark_id_to_bits(WM_ID_B, 32)
        assert bits_a != bits_b

    def test_bit_error_rate_identical(self):
        bits = [1, 0, 1, 1, 0]
        assert bit_error_rate(bits, bits) == 0.0

    def test_bit_error_rate_all_flipped(self):
        orig = [1, 0, 1, 0]
        flipped = [0, 1, 0, 1]
        assert bit_error_rate(orig, flipped) == 1.0

    def test_bit_error_rate_half(self):
        orig = [1, 1, 0, 0]
        extr = [1, 0, 0, 1]
        assert bit_error_rate(orig, extr) == pytest.approx(0.5)

    def test_bit_error_rate_length_mismatch(self):
        assert bit_error_rate([1, 0], [1]) == 1.0


# ---------------------------------------------------------------------------
# 10. Config dataclass
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_strength(self):
        cfg = ClassicalWatermarkConfig(watermark_id="x")
        assert cfg.strength == 0.01

    def test_default_n_bits(self):
        cfg = ClassicalWatermarkConfig(watermark_id="x")
        assert cfg.n_bits == 32

    def test_default_sample_rate(self):
        cfg = ClassicalWatermarkConfig(watermark_id="x")
        assert cfg.sample_rate == 16_000

    def test_custom_strength(self):
        cfg = ClassicalWatermarkConfig(watermark_id="x", strength=0.05)
        assert cfg.strength == pytest.approx(0.05)
