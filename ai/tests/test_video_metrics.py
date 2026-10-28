"""Unit tests for video quality metrics and pipeline evaluation."""
import pytest
import torch

from echoself_ai.video.metrics import (
    compute_temporal_flicker_score,
    compute_video_psnr,
    compute_video_ssim,
    evaluate_video_pipeline,
)
from echoself_ai.video.pipeline import PromptToVideoConfig, PromptToVideoGenerator


def test_video_psnr():
    v1 = torch.zeros(4, 3, 16, 16)
    v2 = torch.zeros(4, 3, 16, 16)
    assert compute_video_psnr(v1, v2) == 100.0

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError):
        compute_video_psnr(v1, torch.zeros(3, 3, 16, 16))


def test_video_ssim():
    v = torch.randn(4, 3, 16, 16).clamp(-1.0, 1.0)
    score = compute_video_ssim(v, v)
    assert score > 0.999

    with pytest.raises(ValueError):
        compute_video_ssim(v, torch.zeros(2, 3, 16, 16))


def test_temporal_flicker_score():
    # Constant progression (linear ramp) has zero flicker variance
    linear = torch.zeros(5, 3, 8, 8)
    for t in range(5):
        linear[t].fill_(t * 0.2)
    score_linear = compute_temporal_flicker_score(linear)
    assert score_linear < 1e-5

    # High flicker: alternating frames
    flicker = torch.zeros(5, 3, 8, 8)
    for t in range(5):
        flicker[t].fill_(1.0 if t % 2 == 0 else -1.0)
    score_flicker = compute_temporal_flicker_score(flicker)
    assert score_flicker >= 0.0

    # Short video (< 3 frames) returns 0.0
    assert compute_temporal_flicker_score(torch.zeros(2, 3, 8, 8)) == 0.0


def test_evaluate_video_pipeline():
    config = PromptToVideoConfig(
        prompt_embed_dim=8,
        identity_embed_dim=8,
        base_dim=8,
        image_size=32,
    )
    generator = PromptToVideoGenerator(config)

    metrics = evaluate_video_pipeline(
        generator=generator,
        test_prompts=["calm sea", "bright stars"],
        num_frames=4,
        device="cpu",
    )

    assert metrics["num_videos_generated"] == 2.0
    assert metrics["total_frames_generated"] == 8.0
    assert metrics["throughput_fps"] > 0.0
    assert metrics["latency_ms_per_video"] > 0.0
    assert metrics["latency_ms_per_frame"] > 0.0
    assert "mean_temporal_loss" in metrics
    assert "mean_flicker_score" in metrics
