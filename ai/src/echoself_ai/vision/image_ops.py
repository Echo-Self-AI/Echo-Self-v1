from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
import torch


@dataclass
class ImageConfig:
    """Image processing configuration and normalization parameters."""

    target_size: tuple[int, int] = (128, 128)
    max_file_size_bytes: int = 15 * 1024 * 1024  # 15 MB safeguard
    normalize_range: tuple[float, float] = (-1.0, 1.0)


def load_image(
    path: str | Path,
    config: Optional[ImageConfig] = None,
) -> torch.Tensor:
    """Load an image file from disk and return a normalized PyTorch tensor of shape (3, H, W)."""
    cfg = config or ImageConfig()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {path}")

    file_size = path.stat().st_size
    if file_size > cfg.max_file_size_bytes:
        raise ValueError(
            f"Image file size {file_size} exceeds maximum allowable limit {cfg.max_file_size_bytes} bytes"
        )

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        resized = img_rgb.resize(cfg.target_size, Image.Resampling.BILINEAR)

    # Convert PIL Image to PyTorch float32 tensor
    # shape: (H, W, 3) in [0, 255]
    tensor = torch.from_numpy(
        __import__("numpy").array(resized, dtype=__import__("numpy").float32)
    )
    # Permute to (3, H, W) and scale to [0.0, 1.0]
    tensor = tensor.permute(2, 0, 1) / 255.0

    # Scale to [-1.0, 1.0] if requested
    if cfg.normalize_range == (-1.0, 1.0):
        tensor = tensor * 2.0 - 1.0

    return tensor


def tensor_to_pil(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """Convert a PyTorch tensor (3, H, W) to a PIL Image."""
    t = tensor.detach().cpu().clone()
    if t.dim() == 4 and t.size(0) == 1:
        t = t.squeeze(0)

    if t.dim() != 3 or t.size(0) != 3:
        raise ValueError(f"Expected tensor of shape (3, H, W), got {t.shape}")

    if denormalize:
        # Map from [-1.0, 1.0] to [0.0, 1.0]
        t = (t + 1.0) / 2.0

    t = torch.clamp(t, 0.0, 1.0) * 255.0
    arr = t.permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="RGB")


def save_image(
    path: str | Path,
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> None:
    """Save a PyTorch tensor (3, H, W) to disk as PNG or JPEG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor_to_pil(tensor, denormalize=denormalize)
    img.save(path)


def generate_synthetic_image(
    width: int = 128,
    height: int = 128,
    pattern: str = "checkerboard",
) -> torch.Tensor:
    """Generate deterministic synthetic RGB images for testing without external assets.

    Returns:
        Tensor of shape (3, height, width) with values in [-1.0, 1.0].
    """
    if pattern == "checkerboard":
        grid_size = 16
        y_coords = torch.arange(height).view(-1, 1) // grid_size
        x_coords = torch.arange(width).view(1, -1) // grid_size
        mask = (x_coords + y_coords) % 2 == 0

        img = torch.zeros(3, height, width, dtype=torch.float32)
        # Channel 0: primary check
        img[0] = torch.where(mask, 0.9, -0.9)
        # Channel 1: inverted check
        img[1] = torch.where(mask, -0.9, 0.9)
        # Channel 2: constant background
        img[2] = 0.2
        return img

    elif pattern == "gradient":
        gx = torch.linspace(-1.0, 1.0, width).view(1, -1).expand(height, -1)
        gy = torch.linspace(-1.0, 1.0, height).view(-1, 1).expand(-1, width)
        img = torch.stack([gx, gy, (gx + gy) / 2.0], dim=0)
        return img

    else:
        # Flat solid neutral
        return torch.zeros(3, height, width, dtype=torch.float32)
