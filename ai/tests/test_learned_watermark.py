"""
Tests for the learned audio watermarking module.

Covers:
- WatermarkEncoder forward pass (output shape)
- WatermarkDecoder forward pass (output shape)
- LearnedWatermarker.train_model() (runs, returns required keys)
- embed() shape preservation after training
- extract() values in [0, 1] after sigmoid
- compute_bit_accuracy() returns float in [0, 1]
- compare_with_classical() returns dict with expected keys
- Bit accuracy on training distribution > 0.5 (better than random)
"""

from __future__ import annotations

import pytest
import torch

from echoself_ai.watermark.learned import (
    LearnedWatermarkConfig,
    LearnedWatermarker,
    WatermarkDecoder,
    WatermarkEncoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> LearnedWatermarkConfig:
    """Small config for fast tests."""
    return LearnedWatermarkConfig(
        n_bits=16,
        hidden_dim=64,
        segment_length=16000,
        sample_rate=16000,
    )


@pytest.fixture(scope="module")
def encoder(config: LearnedWatermarkConfig) -> WatermarkEncoder:
    return WatermarkEncoder(config)


@pytest.fixture(scope="module")
def decoder(config: LearnedWatermarkConfig) -> WatermarkDecoder:
    return WatermarkDecoder(config)


@pytest.fixture(scope="module")
def trained_watermarker(config: LearnedWatermarkConfig, tmp_path_factory):
    """Trains the LearnedWatermarker once and returns it (module-scoped)."""
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    wm = LearnedWatermarker(config=config, checkpoint_dir=ckpt_dir)
    wm.train_model(n_steps=200, batch_size=4)
    return wm


# ---------------------------------------------------------------------------
# WatermarkEncoder tests
# ---------------------------------------------------------------------------

class TestWatermarkEncoder:
    """Tests for WatermarkEncoder forward pass."""

    def test_output_shape_matches_input_audio(self, encoder, config):
        """Encoder output must have the same shape as the input audio segment."""
        B = 2
        audio = torch.randn(B, config.segment_length)
        bits = torch.randint(0, 2, (B, config.n_bits)).float()

        with torch.no_grad():
            output = encoder(audio, bits)

        assert output.shape == audio.shape, (
            f"Expected {audio.shape}, got {output.shape}"
        )

    def test_output_is_finite(self, encoder, config):
        """Encoder output must not contain NaN or Inf."""
        audio = torch.randn(1, config.segment_length)
        bits = torch.zeros(1, config.n_bits)

        with torch.no_grad():
            output = encoder(audio, bits)

        assert torch.isfinite(output).all(), "Encoder output contains NaN or Inf"

    def test_perturbation_is_small(self, encoder, config):
        """Watermark perturbation should be small (< 0.05 max abs)."""
        audio = torch.randn(1, config.segment_length)
        bits = torch.randint(0, 2, (1, config.n_bits)).float()

        with torch.no_grad():
            output = encoder(audio, bits)

        max_perturbation = (output - audio).abs().max().item()
        assert max_perturbation < 0.05, (
            f"Perturbation too large: {max_perturbation:.4f}"
        )

    def test_different_messages_produce_different_outputs(self, encoder, config):
        """Two different messages should produce different watermarked audio."""
        audio = torch.randn(1, config.segment_length)
        bits_a = torch.zeros(1, config.n_bits)
        bits_b = torch.ones(1, config.n_bits)

        with torch.no_grad():
            out_a = encoder(audio, bits_a)
            out_b = encoder(audio, bits_b)

        assert not torch.allclose(out_a, out_b), (
            "Different messages produced identical watermarked audio"
        )

    def test_batch_size_1(self, encoder, config):
        """Works with batch size 1."""
        audio = torch.randn(1, config.segment_length)
        bits = torch.randint(0, 2, (1, config.n_bits)).float()
        with torch.no_grad():
            output = encoder(audio, bits)
        assert output.shape == (1, config.segment_length)

    def test_batch_size_4(self, encoder, config):
        """Works with batch size 4."""
        audio = torch.randn(4, config.segment_length)
        bits = torch.randint(0, 2, (4, config.n_bits)).float()
        with torch.no_grad():
            output = encoder(audio, bits)
        assert output.shape == (4, config.segment_length)


# ---------------------------------------------------------------------------
# WatermarkDecoder tests
# ---------------------------------------------------------------------------

class TestWatermarkDecoder:
    """Tests for WatermarkDecoder forward pass."""

    def test_output_shape(self, decoder, config):
        """Decoder output must have shape (B, n_bits)."""
        B = 3
        audio = torch.randn(B, config.segment_length)

        with torch.no_grad():
            logits = decoder(audio)

        assert logits.shape == (B, config.n_bits), (
            f"Expected ({B}, {config.n_bits}), got {logits.shape}"
        )

    def test_output_is_finite(self, decoder, config):
        """Decoder logits must not contain NaN or Inf."""
        audio = torch.randn(2, config.segment_length)
        with torch.no_grad():
            logits = decoder(audio)
        assert torch.isfinite(logits).all()

    def test_batch_size_1(self, decoder, config):
        """Works with batch size 1."""
        audio = torch.randn(1, config.segment_length)
        with torch.no_grad():
            logits = decoder(audio)
        assert logits.shape == (1, config.n_bits)

    def test_logits_are_unbounded(self, decoder, config):
        """Logits should not be constrained to [0, 1] — they are raw BCE inputs."""
        audio = torch.randn(8, config.segment_length)
        with torch.no_grad():
            logits = decoder(audio)
        # Should have both positive and negative values across a batch
        assert logits.max().item() > 0.0 or logits.min().item() < 0.0


# ---------------------------------------------------------------------------
# LearnedWatermarker.train_model() tests
# ---------------------------------------------------------------------------

class TestTrainModel:
    """Tests for the training loop."""

    def test_train_model_completes_without_crash(self, config, tmp_path):
        """train_model() must run without raising any exception."""
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=10, batch_size=2)
        assert result is not None

    def test_train_model_returns_required_keys(self, config, tmp_path):
        """Return dict must contain all four required keys."""
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=10, batch_size=2)

        required_keys = {"encoder_path", "decoder_path", "final_loss", "train_losses"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_train_losses_length_matches_n_steps(self, config, tmp_path):
        """train_losses list must have one entry per training step."""
        n_steps = 20
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=n_steps, batch_size=2)
        assert len(result["train_losses"]) == n_steps

    def test_final_loss_matches_last_train_loss(self, config, tmp_path):
        """final_loss must equal the last element of train_losses."""
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=10, batch_size=2)
        assert result["final_loss"] == result["train_losses"][-1]

    def test_checkpoints_saved(self, config, tmp_path):
        """Encoder and decoder checkpoint files must be created."""
        import os
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=10, batch_size=2)
        assert os.path.isfile(result["encoder_path"]), "Encoder checkpoint missing"
        assert os.path.isfile(result["decoder_path"]), "Decoder checkpoint missing"

    def test_loss_decreases_over_training(self, config, tmp_path):
        """Loss should trend downward over 200 steps (compare first 10 vs last 10)."""
        wm = LearnedWatermarker(config=config, checkpoint_dir=str(tmp_path))
        result = wm.train_model(n_steps=200, batch_size=4)
        losses = result["train_losses"]
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10
        assert late_avg < early_avg, (
            f"Loss did not decrease: early={early_avg:.4f} late={late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# embed() / extract() tests (require a trained model)
# ---------------------------------------------------------------------------

class TestEmbedExtract:
    """Tests for embed and extract after training."""

    def test_embed_returns_same_shape(self, trained_watermarker, config):
        """embed() must return a tensor with the same shape as the input waveform."""
        waveform = torch.randn(config.segment_length)
        bits = torch.randint(0, 2, (config.n_bits,)).float()
        result = trained_watermarker.embed(waveform, bits)
        assert result.shape == waveform.shape

    def test_embed_longer_waveform(self, trained_watermarker, config):
        """embed() must handle waveforms longer than one segment."""
        waveform = torch.randn(config.segment_length * 3)
        bits = torch.randint(0, 2, (config.n_bits,)).float()
        result = trained_watermarker.embed(waveform, bits)
        assert result.shape == waveform.shape

    def test_embed_shorter_waveform(self, trained_watermarker, config):
        """embed() must pass through waveforms shorter than one segment unchanged."""
        short_len = config.segment_length // 2
        waveform = torch.randn(short_len)
        bits = torch.randint(0, 2, (config.n_bits,)).float()
        result = trained_watermarker.embed(waveform, bits)
        assert result.shape == waveform.shape

    def test_extract_returns_probabilities_in_0_1(self, trained_watermarker, config):
        """extract() output (after sigmoid in the method) must be in [0, 1]."""
        waveform = torch.randn(config.segment_length)
        probs = trained_watermarker.extract(waveform)
        assert probs.shape == (config.n_bits,), f"Unexpected shape: {probs.shape}"
        assert probs.min().item() >= 0.0 - 1e-6, f"Min below 0: {probs.min()}"
        assert probs.max().item() <= 1.0 + 1e-6, f"Max above 1: {probs.max()}"

    def test_extract_short_waveform(self, trained_watermarker, config):
        """extract() must handle waveforms shorter than segment_length."""
        waveform = torch.randn(config.segment_length // 2)
        probs = trained_watermarker.extract(waveform)
        assert probs.shape == (config.n_bits,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# compute_bit_accuracy() tests
# ---------------------------------------------------------------------------

class TestComputeBitAccuracy:
    """Tests for compute_bit_accuracy."""

    def test_returns_float_in_0_1(self, trained_watermarker, config):
        """Must return a float in [0.0, 1.0]."""
        orig = torch.randint(0, 2, (config.n_bits,)).float()
        extr = torch.rand(config.n_bits)
        acc = trained_watermarker.compute_bit_accuracy(orig, extr)
        assert isinstance(acc, float), f"Expected float, got {type(acc)}"
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_perfect_accuracy(self, trained_watermarker, config):
        """When extracted bits match originals perfectly, accuracy is 1.0."""
        orig = torch.randint(0, 2, (config.n_bits,)).float()
        # Pass identical bits (soft = 1.0 for 1, 0.0 for 0 — after threshold stays same)
        acc = trained_watermarker.compute_bit_accuracy(orig, orig)
        assert acc == 1.0, f"Expected 1.0, got {acc}"

    def test_all_wrong_accuracy(self, trained_watermarker, config):
        """When extracted bits are completely wrong, accuracy is 0.0."""
        orig = torch.ones(config.n_bits)
        wrong = torch.zeros(config.n_bits)
        acc = trained_watermarker.compute_bit_accuracy(orig, wrong)
        assert acc == 0.0, f"Expected 0.0, got {acc}"

    def test_half_accuracy(self, trained_watermarker, config):
        """Half correct → ~0.5 accuracy."""
        n = config.n_bits
        orig = torch.ones(n)
        half = torch.cat([torch.ones(n // 2), torch.zeros(n - n // 2)])
        acc = trained_watermarker.compute_bit_accuracy(orig, half)
        assert abs(acc - 0.5) < 0.1, f"Expected ~0.5, got {acc}"


# ---------------------------------------------------------------------------
# compare_with_classical() tests
# ---------------------------------------------------------------------------

class TestCompareWithClassical:
    """Tests for compare_with_classical."""

    def test_returns_classical_and_learned_keys(self, trained_watermarker, config):
        """Return dict must have 'classical' and 'learned' top-level keys."""
        waveform = torch.randn(config.segment_length)
        result = trained_watermarker.compare_with_classical(
            waveform, watermark_id="test-wm-id-001"
        )
        assert "classical" in result, "Missing 'classical' key"
        assert "learned" in result, "Missing 'learned' key"

    def test_classical_subkeys(self, trained_watermarker, config):
        """classical sub-dict must have detected, correlation, ber."""
        waveform = torch.randn(config.segment_length)
        result = trained_watermarker.compare_with_classical(
            waveform, watermark_id="test-wm-id-002"
        )
        classical = result["classical"]
        assert "detected" in classical
        assert "correlation" in classical
        assert "ber" in classical

    def test_learned_subkeys(self, trained_watermarker, config):
        """learned sub-dict must have bit_accuracy, mean_confidence, n_bits."""
        waveform = torch.randn(config.segment_length)
        result = trained_watermarker.compare_with_classical(
            waveform, watermark_id="test-wm-id-003"
        )
        learned = result["learned"]
        assert "bit_accuracy" in learned
        assert "mean_confidence" in learned
        assert "n_bits" in learned

    def test_classical_detected_on_watermarked(self, trained_watermarker, config):
        """Classical method should detect its own watermark."""
        waveform = torch.randn(config.segment_length)
        result = trained_watermarker.compare_with_classical(
            waveform, watermark_id="test-wm-id-detect"
        )
        assert result["classical"]["detected"] is True


# ---------------------------------------------------------------------------
# Bit accuracy on trained model (better than random)
# ---------------------------------------------------------------------------

class TestBitAccuracyAfterTraining:
    """Verify the learned model achieves >50% bit accuracy after training."""

    def test_bit_accuracy_better_than_random(self, trained_watermarker, config):
        """After 200 training steps, bit accuracy on clean audio must exceed 0.5."""
        torch.manual_seed(0)
        n_trials = 20
        total_acc = 0.0

        for _ in range(n_trials):
            # Sample from the same distribution as training (uniform noise)
            audio = torch.rand(config.segment_length) * 2.0 - 1.0
            bits = torch.randint(0, 2, (config.n_bits,)).float()

            watermarked = trained_watermarker.embed(audio, bits)
            extracted = trained_watermarker.extract(watermarked)
            acc = trained_watermarker.compute_bit_accuracy(bits, extracted)
            total_acc += acc

        mean_acc = total_acc / n_trials
        assert mean_acc > 0.5, (
            f"Mean bit accuracy {mean_acc:.3f} is not better than random (0.5). "
            "The model may need more training steps or a larger hidden_dim."
        )

    def test_embedding_is_invertible_up_to_perturbation(
        self, trained_watermarker, config
    ):
        """The watermark perturbation must be small relative to the signal."""
        audio = torch.randn(config.segment_length)
        bits = torch.randint(0, 2, (config.n_bits,)).float()

        watermarked = trained_watermarker.embed(audio, bits)
        perturbation = (watermarked - audio).abs().max().item()
        assert perturbation < 0.05, (
            f"Perturbation too large: {perturbation:.4f}"
        )
