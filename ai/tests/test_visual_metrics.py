"""Unit tests for visual metrics (PSNR, SSIM, MSE) and avatar evaluation pipeline."""
import pytest
import torch

from echoself_ai.vision.avatar import AvatarGenerator, AvatarGeneratorConfig
from echoself_ai.vision.encoder import VisualEncoder, VisualEncoderConfig
from echoself_ai.vision.metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    evaluate_avatar_generation,
)


def test_compute_mse():
    t1 = torch.zeros(1, 3, 32, 32)
    t2 = torch.zeros(1, 3, 32, 32)
    assert compute_mse(t1, t2) == 0.0

    t3 = torch.ones(1, 3, 32, 32)
    assert abs(compute_mse(t1, t3) - 1.0) < 1e-6

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError):
        compute_mse(t1, torch.zeros(1, 3, 16, 16))


def test_compute_psnr():
    t1 = torch.zeros(1, 3, 32, 32)
    t2 = torch.zeros(1, 3, 32, 32)
    # Perfect match -> 100 dB cap
    assert compute_psnr(t1, t2) == 100.0

    t3 = torch.ones(1, 3, 32, 32)
    # Dynamic range = 2.0 (domain [-1, 1]), MSE = 1.0 -> 10 * log10(4.0 / 1.0) ~ 6.02 dB
    psnr = compute_psnr(t1, t3, dynamic_range=2.0)
    assert abs(psnr - 6.0205999) < 1e-3


def test_compute_ssim():
    # Identical images have SSIM near 1.0
    img = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    ssim_self = compute_ssim(img, img)
    assert ssim_self > 0.999

    # Highly dissimilar images have lower SSIM
    img_inv = -img
    ssim_inv = compute_ssim(img, img_inv)
    assert ssim_inv < ssim_self


def test_evaluate_avatar_generation():
    enc_cfg = VisualEncoderConfig(embedding_dim=32, base_dim=8)
    gen_cfg = AvatarGeneratorConfig(embedding_dim=32, condition_dim=8, base_dim=8, image_size=32)

    encoder = VisualEncoder(enc_cfg)
    generator = AvatarGenerator(gen_cfg)

    metrics = evaluate_avatar_generation(
        generator=generator,
        encoder=encoder,
        num_samples=4,
        batch_size=2,
        device="cpu",
    )

    assert "throughput_fps" in metrics
    assert "latency_ms_per_image" in metrics
    assert "mean_identity_similarity" in metrics
    assert metrics["num_samples"] == 4.0
    assert metrics["throughput_fps"] > 0.0
    assert metrics["latency_ms_per_image"] > 0.0
