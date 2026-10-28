"""Unit tests for video frame sequences, synthetic generators, and I/O containers."""
from pathlib import Path
import pytest
import torch

from echoself_ai.video.frames import (
    FrameSequence,
    VideoConfig,
    generate_synthetic_video,
)


def test_frame_sequence_validation():
    # Valid 4D tensor (T, C, H, W)
    t = torch.randn(8, 3, 32, 32)
    seq = FrameSequence(t, fps=24)
    assert seq.num_frames == 8
    assert seq.channels == 3
    assert seq.height == 32
    assert seq.width == 32
    assert seq.fps == 24
    assert abs(seq.duration_seconds - (8 / 24)) < 1e-5

    # 5D tensor with singleton batch (1, T, C, H, W) is auto-squeezed
    t5 = torch.randn(1, 4, 3, 16, 16)
    seq5 = FrameSequence(t5)
    assert seq5.num_frames == 4

    # Invalid dimensions raise ValueError
    with pytest.raises(ValueError):
        FrameSequence(torch.randn(2, 4, 3, 16, 16))  # batch > 1
    with pytest.raises(ValueError):
        FrameSequence(torch.randn(3, 16, 16))  # 3D
    with pytest.raises(ValueError):
        FrameSequence(torch.randn(4, 5, 16, 16))  # 5 channels


def test_frame_sequence_indexing_and_iteration():
    t = torch.randn(10, 3, 16, 16)
    seq = FrameSequence(t, fps=10)

    assert len(seq) == 10
    # Single index returns 3D frame tensor
    frame0 = seq[0]
    assert frame0.shape == (3, 16, 16)

    # Slice returns child FrameSequence
    sub_seq = seq[2:6]
    assert isinstance(sub_seq, FrameSequence)
    assert sub_seq.num_frames == 4
    assert sub_seq.fps == 10

    # Iteration
    count = 0
    for f in seq:
        assert f.shape == (3, 16, 16)
        count += 1
    assert count == 10


def test_synthetic_video_generation():
    seq_bounce = generate_synthetic_video(num_frames=8, width=32, height=32, pattern="bounce")
    assert seq_bounce.num_frames == 8
    assert seq_bounce.frames.min() >= -1.0
    assert seq_bounce.frames.max() <= 1.0

    # Check temporal difference is non-zero (there is motion)
    diff = (seq_bounce.frames[1] - seq_bounce.frames[0]).abs().sum()
    assert diff > 0.0

    seq_sweep = generate_synthetic_video(num_frames=6, width=16, height=16, pattern="sweep")
    assert seq_sweep.num_frames == 6
    diff_sweep = (seq_sweep.frames[-1] - seq_sweep.frames[0]).abs().sum()
    assert diff_sweep > 0.0

    with pytest.raises(ValueError):
        generate_synthetic_video(pattern="unknown")


def test_gif_io_roundtrip(tmp_path: Path):
    seq = generate_synthetic_video(num_frames=5, width=24, height=24, fps=10, pattern="bounce")
    gif_path = tmp_path / "test_animation.gif"

    saved_path = seq.save_gif(gif_path)
    assert saved_path.exists()

    loaded = FrameSequence.load_gif(gif_path, fps=10)
    assert loaded.num_frames == 5
    assert loaded.height == 24
    assert loaded.width == 24
    assert loaded.fps == 10


def test_frame_directory_io_roundtrip(tmp_path: Path):
    seq = generate_synthetic_video(num_frames=4, width=20, height=20, fps=12, pattern="sweep")
    frames_dir = tmp_path / "frames"

    seq.save_frame_directory(frames_dir, prefix="seq")
    assert (frames_dir / "seq_00000.png").exists()
    assert (frames_dir / "seq_00003.png").exists()

    loaded = FrameSequence.load_frame_directory(frames_dir, fps=12, pattern="seq_*.png")
    assert loaded.num_frames == 4
    assert loaded.height == 20
    assert loaded.width == 20
    assert loaded.fps == 12
