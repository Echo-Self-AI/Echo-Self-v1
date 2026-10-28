"""Image file I/O, format validation, dimension checks, and normalization."""
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from echoself_ai.voice.audio import MediaValidationError


# -----------------------------------------------------------------------------
# Validation Error Taxonomy
# -----------------------------------------------------------------------------

class ImageValidationError(MediaValidationError, ValueError):
    """Base exception for image-specific validation failures."""
    pass


class InvalidImageFormatError(ImageValidationError):
    """Raised when an image container or encoding is unsupported."""
    pass


class ImageDimensionError(ImageValidationError):
    """Raised when image dimensions are out of allowable bounds."""
    pass


class ImageFileSizeError(ImageValidationError):
    """Raised when an image file exceeds size thresholds."""
    pass


class CorruptImageError(ImageValidationError):
    """Raised when an image file fails decoding or contains corrupt data."""
    pass


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ImageConfig:
    """Image processing configuration and normalization parameters."""

    target_size: tuple[int, int] = (128, 128)
    min_dimension: int = 32
    max_dimension: int = 4096
    max_file_size_bytes: int = 15 * 1024 * 1024  # 15 MB safeguard
    normalize_range: tuple[float, float] = (-1.0, 1.0)
    allowed_formats: tuple[str, ...] = ("PNG", "JPEG", "JPG", "WEBP")



# -----------------------------------------------------------------------------
# Validation & Ingestion
# -----------------------------------------------------------------------------

def ingest_image(
    source: Union[str, Path, bytes, BytesIO],
    config: Optional[ImageConfig] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Ingest, validate, downsample, and normalize an input image.

    Accepts file paths, raw bytes, or BytesIO objects.
    Validates file size, container format (PNG/JPEG/WebP), dimensions, and finite values.

    Returns:
        tensor: Normalized PyTorch float32 tensor of shape (3, target_H, target_W) in [-1.0, 1.0].
    """
    cfg = config or ImageConfig()
    out_size = target_size or cfg.target_size

    # Check file size before opening
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found at: {p}")
        file_size = p.stat().st_size
        if file_size > cfg.max_file_size_bytes:
            raise ImageFileSizeError(
                f"Image file size {file_size} exceeds maximum allowable limit {cfg.max_file_size_bytes} bytes"
            )
        if file_size == 0:
            raise CorruptImageError("Image file is empty (0 bytes)")
        bio = open(p, "rb")
        needs_close = True
    elif isinstance(source, bytes):
        if len(source) > cfg.max_file_size_bytes:
            raise ImageFileSizeError(
                f"Image byte buffer size {len(source)} exceeds limit {cfg.max_file_size_bytes} bytes"
            )
        if len(source) == 0:
            raise CorruptImageError("Image byte buffer is empty (0 bytes)")
        bio = BytesIO(source)
        needs_close = False
    elif isinstance(source, BytesIO):
        source.seek(0, 2)
        size = source.tell()
        source.seek(0)
        if size > cfg.max_file_size_bytes:
            raise ImageFileSizeError(
                f"Image stream size {size} exceeds limit {cfg.max_file_size_bytes} bytes"
            )
        if size == 0:
            raise CorruptImageError("Image stream is empty (0 bytes)")
        bio = source
        needs_close = False
    else:
        raise InvalidImageFormatError(f"Unsupported image source type: {type(source).__name__}")

    try:
        try:
            with Image.open(bio) as img:
                fmt = (img.format or "").upper()
                if fmt not in cfg.allowed_formats:
                    raise InvalidImageFormatError(
                        f"Unsupported image format: '{fmt}'. Allowed formats are: {cfg.allowed_formats}"
                    )

                w, h = img.size
                if w < cfg.min_dimension or h < cfg.min_dimension:
                    raise ImageDimensionError(
                        f"Image dimensions ({w}x{h}) smaller than minimum allowed {cfg.min_dimension}x{cfg.min_dimension}"
                    )
                if w > cfg.max_dimension or h > cfg.max_dimension:
                    raise ImageDimensionError(
                        f"Image dimensions ({w}x{h}) exceed maximum allowed {cfg.max_dimension}x{cfg.max_dimension}"
                    )

                img_rgb = img.convert("RGB")
                resized = img_rgb.resize(out_size, Image.Resampling.BILINEAR)

                # Convert to numpy array
                arr = np.array(resized, dtype=np.float32)

        except (UnidentifiedImageError, OSError, IOError) as exc:
            if isinstance(exc, (ImageValidationError, FileNotFoundError)):
                raise
            raise CorruptImageError(f"Failed to decode image data: {exc}") from exc

        # Convert to PyTorch float32 tensor
        # shape: (H, W, 3) in [0.0, 255.0]
        tensor = torch.from_numpy(arr)
        # Permute to (3, H, W) and scale to [0.0, 1.0]
        tensor = tensor.permute(2, 0, 1) / 255.0

        if not torch.isfinite(tensor).all():
            raise CorruptImageError("Image tensor contains non-finite (NaN or Inf) pixel values")

        # Scale to [-1.0, 1.0] if requested
        if cfg.normalize_range == (-1.0, 1.0):
            tensor = tensor * 2.0 - 1.0

        return tensor

    finally:
        if needs_close:
            bio.close()


def load_image(
    path: Union[str, Path],
    config: Optional[ImageConfig] = None,
) -> torch.Tensor:
    """Load an image file from disk and return a normalized PyTorch tensor of shape (3, H, W)."""
    cfg = config or ImageConfig(target_size=(128, 128), min_dimension=1)
    return ingest_image(path, config=cfg, target_size=cfg.target_size)


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
    path: Union[str, Path],
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
