"""Unit tests for temporal dynamics, consistency loss, and neural frame interpolation."""
import pytest
import torch

from echoself_ai.video.frames import FrameSequence, generate_synthetic_video
from echoself_ai.video.temporal import (
    FrameInterpolator,
    FrameInterpolatorConfig,
    TemporalConvBlock,
    compute_motion_magnitude,
    compute_temporal_consistency_loss,
)


def test_temporal_consistency_loss():
    # 1. Static video has 0 temporal loss
    static_frames = torch.ones(2, 6, 3, 16, 16) * 0.5
    loss_static = compute_temporal_consistency_loss(static_frames)
    assert abs(loss_static.item()) < 1e-6

    # 2. Smooth video has low loss
    smooth_video = generate_synthetic_video(num_frames=8, width=16, height=16, pattern="sweep")
    loss_smooth = compute_temporal_consistency_loss(smooth_video.frames)

    # 3. Random noise video has much higher temporal loss
    noise_frames = torch.randn(8, 3, 16, 16).clamp(-1.0, 1.0)
    loss_noise = compute_temporal_consistency_loss(noise_frames)

    assert loss_noise > loss_smooth
    assert loss_smooth > loss_static

    # 4. Short video (< 2 frames) returns 0
    single_frame = torch.randn(1, 3, 16, 16)
    assert compute_temporal_consistency_loss(single_frame).item() == 0.0


def test_motion_magnitude():
    static_frames = torch.zeros(1, 4, 3, 16, 16)
    mag_static = compute_motion_magnitude(static_frames)
    assert mag_static.shape == (1, 3, 1, 16, 16)
    assert mag_static.max().item() < 1e-4

    sweep_video = generate_synthetic_video(num_frames=5, width=16, height=16, pattern="sweep")
    mag_sweep = compute_motion_magnitude(sweep_video.frames)
    assert mag_sweep.shape == (1, 4, 1, 16, 16)
    assert mag_sweep.mean().item() > 0.0


def test_temporal_conv_block():
    block = TemporalConvBlock(in_channels=4, out_channels=8, kernel_size=3)
    x = torch.randn(2, 6, 4, 16, 16)  # (B, T, C, H, W)
    out = block(x)

    assert out.shape == (2, 6, 8, 16, 16)

    # Test backpropagation
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in block.parameters())


def test_frame_interpolator_forward():
    config = FrameInterpolatorConfig(base_dim=8)
    interpolator = FrameInterpolator(config)

    frame_a = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    frame_b = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)

    # Midpoint alpha = 0.5
    out = interpolator(frame_a, frame_b, alpha=0.5)
    assert out.shape == (2, 3, 32, 32)
    assert out.min() >= -1.0
    assert out.max() <= 1.0

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in interpolator.parameters())


def test_frame_interpolator_sequence():
    config = FrameInterpolatorConfig(base_dim=8)
    interpolator = FrameInterpolator(config)

    frame_start = torch.ones(3, 16, 16) * -0.8
    frame_end = torch.ones(3, 16, 16) * 0.8

    seq = interpolator.interpolate_sequence(
        start_frame=frame_start,
        end_frame=frame_end,
        num_intermediate_frames=4,
        fps=10,
    )

    assert isinstance(seq, FrameSequence)
    # Total frames = 1 (start) + 4 (intermediate) + 1 (end) = 6
    assert seq.num_frames == 6
    assert seq.fps == 10
    assert seq.frames.shape == (6, 3, 16, 16)
