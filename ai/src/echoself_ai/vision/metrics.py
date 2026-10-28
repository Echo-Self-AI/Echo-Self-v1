"""Visual quality evaluation metrics and avatar benchmarking pipeline."""
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from echoself_ai.vision.avatar import AvatarGenerator
from echoself_ai.vision.encoder import VisualEncoder


def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute mean squared error between two image tensors."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    return float(F.mse_loss(img1, img2).item())


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, dynamic_range: float = 2.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        img1: Image tensor in range [-1.0, 1.0].
        img2: Image tensor in range [-1.0, 1.0].
        dynamic_range: Peak-to-peak signal value (2.0 for range [-1.0, 1.0]).

    Returns:
        PSNR value in dB. Returns 100.0 dB if MSE is effectively zero.
    """
    mse = compute_mse(img1, img2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * math.log10((dynamic_range**2) / mse))


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 7,
    dynamic_range: float = 2.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) using PyTorch tensor operations.

    Args:
        img1: Image tensor of shape (B, C, H, W) in range [-1.0, 1.0].
        img2: Image tensor of shape (B, C, H, W) in range [-1.0, 1.0].
        window_size: Spatial pooling kernel size.
        dynamic_range: Value range of pixel domain (2.0 for [-1.0, 1.0]).

    Returns:
        Mean SSIM across all channels and spatial locations.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    # Uniform averaging kernel
    channels = img1.shape[1]
    kernel = torch.ones((channels, 1, window_size, window_size), device=img1.device) / (
        window_size * window_size
    )

    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


def evaluate_avatar_generation(
    generator: AvatarGenerator,
    encoder: VisualEncoder,
    num_samples: int = 16,
    batch_size: int = 4,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark avatar generation speed, identity fidelity, and memory efficiency.

    Args:
        generator: AvatarGenerator instance.
        encoder: VisualEncoder instance for identity similarity checking.
        num_samples: Total number of test avatar images to generate.
        batch_size: Evaluation batch size.
        device: Target execution device ("cpu" or "cuda").

    Returns:
        Dictionary of metrics including latency, FPS, and identity preservation.
    """
    generator.eval()
    encoder.eval()

    dev = torch.device(device)
    generator.to(dev)
    encoder.to(dev)

    embedding_dim = generator.config.embedding_dim
    condition_dim = generator.config.condition_dim

    total_time = 0.0
    identity_similarities: list[float] = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_bs = min(batch_size, num_samples - i * batch_size)
            # Create normalized reference identity embeddings
            target_embs = torch.randn(current_bs, embedding_dim, device=dev)
            target_embs = F.normalize(target_embs, p=2, dim=-1)

            conds = torch.randn(current_bs, condition_dim, device=dev)

            t0 = time.perf_counter()
            gen_images = generator(target_embs, conds)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            # Measure identity preservation
            extracted_embs = encoder(gen_images)
            cos_sim = F.cosine_similarity(extracted_embs, target_embs, dim=-1)
            identity_similarities.extend(cos_sim.cpu().tolist())

    mean_identity_sim = (
        sum(identity_similarities) / len(identity_similarities) if identity_similarities else 0.0
    )
    fps = num_samples / total_time if total_time > 0 else 0.0
    latency_ms = (total_time / num_samples) * 1000.0 if num_samples > 0 else 0.0

    return {
        "num_samples": float(num_samples),
        "total_time_seconds": float(total_time),
        "latency_ms_per_image": float(latency_ms),
        "throughput_fps": float(fps),
        "mean_identity_similarity": float(mean_identity_sim),
    }
