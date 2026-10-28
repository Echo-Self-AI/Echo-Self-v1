"""Video quality metrics, temporal stability scoring, and pipeline benchmarking."""
import time
from typing import Optional

import torch
import torch.nn.functional as F

from echoself_ai.video.frames import FrameSequence
from echoself_ai.video.pipeline import PromptToVideoGenerator
from echoself_ai.video.temporal import compute_temporal_consistency_loss
from echoself_ai.vision.metrics import compute_psnr, compute_ssim


def compute_video_psnr(
    video1: torch.Tensor,
    video2: torch.Tensor,
    dynamic_range: float = 2.0,
) -> float:
    """Compute average frame-wise Peak Signal-to-Noise Ratio (PSNR) across two video tensors.

    Args:
        video1: Tensor of shape (T, C, H, W) or (B, T, C, H, W).
        video2: Tensor of shape (T, C, H, W) or (B, T, C, H, W).
        dynamic_range: Peak-to-peak signal value (2.0 for range [-1.0, 1.0]).

    Returns:
        Average PSNR across all frames in dB.
    """
    if video1.dim() == 5:
        video1 = video1.squeeze(0)
    if video2.dim() == 5:
        video2 = video2.squeeze(0)

    if video1.shape != video2.shape:
        raise ValueError(f"Shape mismatch: {video1.shape} vs {video2.shape}")

    t = video1.shape[0]
    psnr_scores: list[float] = []
    for i in range(t):
        score = compute_psnr(video1[i], video2[i], dynamic_range=dynamic_range)
        psnr_scores.append(score)

    return float(sum(psnr_scores) / len(psnr_scores)) if psnr_scores else 0.0


def compute_video_ssim(
    video1: torch.Tensor,
    video2: torch.Tensor,
    window_size: int = 7,
) -> float:
    """Compute average frame-wise Structural Similarity Index (SSIM) across two video tensors."""
    if video1.dim() == 5:
        video1 = video1.squeeze(0)
    if video2.dim() == 5:
        video2 = video2.squeeze(0)

    if video1.shape != video2.shape:
        raise ValueError(f"Shape mismatch: {video1.shape} vs {video2.shape}")

    t = video1.shape[0]
    ssim_scores: list[float] = []
    for i in range(t):
        score = compute_ssim(video1[i], video2[i], window_size=window_size)
        ssim_scores.append(score)

    return float(sum(ssim_scores) / len(ssim_scores)) if ssim_scores else 0.0


def compute_temporal_flicker_score(frames: torch.Tensor) -> float:
    """Compute temporal variance/flicker score across video frames.

    Calculates standard deviation of frame-to-frame L1 differences across time.
    Lower score indicates uniform, smooth transitions without sudden flickering.
    """
    if frames.dim() == 5:
        frames = frames.squeeze(0)

    t = frames.shape[0]
    if t < 3:
        return 0.0

    diffs: list[float] = []
    for i in range(1, t):
        l1_diff = (frames[i] - frames[i - 1]).abs().mean().item()
        diffs.append(l1_diff)

    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    return float(variance**0.5)


def evaluate_video_pipeline(
    generator: PromptToVideoGenerator,
    test_prompts: Optional[list[str]] = None,
    num_frames: int = 8,
    fps: int = 16,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark prompt-to-video generation latency, throughput, and stability.

    Args:
        generator: PromptToVideoGenerator instance.
        test_prompts: List of prompt strings to test.
        num_frames: Number of frames per synthesized video sequence.
        fps: Target frames per second.
        device: Execution device ("cpu" or "cuda").

    Returns:
        Dictionary of metrics including generation latency, FPS, and temporal consistency.
    """
    dev = torch.device(device)
    generator.to(dev)
    generator.eval()

    prompts = test_prompts or [
        "avatar speaking calmly",
        "gentle head turn with smile",
        "blinking with ambient background",
    ]

    total_time = 0.0
    total_frames = 0
    temporal_losses: list[float] = []
    flicker_scores: list[float] = []

    with torch.no_grad():
        for prompt in prompts:
            t0 = time.perf_counter()
            video_tensor = generator(prompt=prompt, num_frames=num_frames)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            total_frames += num_frames
            t_loss = compute_temporal_consistency_loss(video_tensor).item()
            temporal_losses.append(t_loss)
            flicker = compute_temporal_flicker_score(video_tensor)
            flicker_scores.append(flicker)

    num_videos = len(prompts)
    fps_throughput = total_frames / total_time if total_time > 0 else 0.0
    latency_ms_per_video = (total_time / num_videos) * 1000.0 if num_videos > 0 else 0.0
    latency_ms_per_frame = (total_time / total_frames) * 1000.0 if total_frames > 0 else 0.0

    return {
        "num_videos_generated": float(num_videos),
        "total_frames_generated": float(total_frames),
        "total_time_seconds": float(total_time),
        "latency_ms_per_video": float(latency_ms_per_video),
        "latency_ms_per_frame": float(latency_ms_per_frame),
        "throughput_fps": float(fps_throughput),
        "mean_temporal_loss": float(sum(temporal_losses) / len(temporal_losses)),
        "mean_flicker_score": float(sum(flicker_scores) / len(flicker_scores)),
    }
