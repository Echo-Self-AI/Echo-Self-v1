"""Portrait image processor for EchoSelf avatar pipeline.

This module processes a user-supplied portrait photograph and produces a
clean, standardised avatar image suitable for display and for use as the
anchor frame in talking-head video generation.

What it does (honest description):
- Validates that the supplied image is a recognisable portrait (non-trivial
  content, minimum resolution).
- Crops and centres the image to a square aspect ratio.
- Resizes to the requested output resolution (default 512×512).
- Applies a mild bilateral-style filter to reduce compression artefacts.
- Enhances contrast and colour saturation slightly for a polished look.
- Saves the processed image as a PNG alongside the original.

What it does NOT do:
- It does not use a generative AI model to transform the portrait.
- It does not remove backgrounds, replace faces, or alter identity.
- It does not claim to be more than image processing.

All operations use Pillow (PIL). No neural network is involved.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class PortraitProcessorConfig:
    output_size: int = 512          # Output square side length in pixels
    min_input_size: int = 64        # Minimum acceptable input dimension
    max_input_bytes: int = 20 * 1024 * 1024  # 20 MB
    enhance_contrast: float = 1.15  # Mild contrast boost (1.0 = unchanged)
    enhance_color: float = 1.10     # Mild saturation boost
    enhance_sharpness: float = 1.20 # Mild sharpness boost
    storage_dir: str = "data/avatars"


@dataclass
class PortraitResult:
    avatar_id: str
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    output_path: str
    thumbnail_path: str
    image_base64: str           # PNG base64 of the processed avatar
    thumbnail_base64: str       # PNG base64 of a 128×128 thumbnail
    processing_method: str      # Always "pillow-portrait-processing"
    sha256: str
    created_at: str


class PortraitProcessor:
    """Processes a portrait photograph into a standardised avatar image."""

    def __init__(self, config: Optional[PortraitProcessorConfig] = None) -> None:
        self.config = config or PortraitProcessorConfig()
        self._storage = Path(self.config.storage_dir)
        self._storage.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_bytes: bytes, user_id: str, avatar_id: str) -> PortraitResult:
        """Process raw image bytes into a standardised avatar.

        Args:
            image_bytes: Raw bytes of a JPEG, PNG, WEBP, or BMP image.
            user_id: User identifier used for storage organisation.
            avatar_id: Identifier for this avatar asset.

        Returns:
            PortraitResult with paths and base64-encoded output.

        Raises:
            ValueError: If image is invalid, too small, or too large.
        """
        self._validate_input(image_bytes)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = img.size
        self._validate_dimensions(img)

        processed = self._process(img)
        thumbnail = processed.resize((128, 128), Image.LANCZOS)

        # Compute hash of the processed image
        proc_bytes = self._to_png_bytes(processed)
        sha256 = hashlib.sha256(proc_bytes).hexdigest()

        # Storage paths
        user_dir = self._storage / _safe_id(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        safe_avatar = _safe_id(avatar_id)
        out_path = user_dir / f"{safe_avatar}.png"
        thumb_path = user_dir / f"{safe_avatar}_thumb.png"

        # Persist
        processed.save(out_path, format="PNG", optimize=True)
        thumbnail.save(thumb_path, format="PNG", optimize=True)

        # Also save original
        orig_path = user_dir / f"{safe_avatar}_original.jpg"
        try:
            orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            orig_img.save(orig_path, format="JPEG", quality=92)
        except Exception:
            pass

        # Save metadata
        meta = {
            "user_id": user_id,
            "avatar_id": avatar_id,
            "original_size": list(original_size),
            "output_size": list(processed.size),
            "sha256": sha256,
            "created_at": _now_iso(),
            "processing_method": "pillow-portrait-processing",
            "output_path": str(out_path),
            "thumbnail_path": str(thumb_path),
        }
        (user_dir / f"{safe_avatar}_meta.json").write_text(json.dumps(meta, indent=2))

        return PortraitResult(
            avatar_id=avatar_id,
            original_size=original_size,
            output_size=processed.size,
            output_path=str(out_path),
            thumbnail_path=str(thumb_path),
            image_base64=_to_b64(proc_bytes),
            thumbnail_base64=_to_b64(self._to_png_bytes(thumbnail)),
            processing_method="pillow-portrait-processing",
            sha256=sha256,
            created_at=_now_iso(),
        )

    def load(self, user_id: str, avatar_id: str) -> Optional[PortraitResult]:
        """Load a previously processed avatar from storage."""
        user_dir = self._storage / _safe_id(user_id)
        safe_avatar = _safe_id(avatar_id)
        meta_path = user_dir / f"{safe_avatar}_meta.json"
        out_path = user_dir / f"{safe_avatar}.png"
        thumb_path = user_dir / f"{safe_avatar}_thumb.png"

        if not meta_path.exists() or not out_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            img_bytes = self._to_png_bytes(Image.open(out_path).convert("RGB"))
            thumb_bytes = self._to_png_bytes(Image.open(thumb_path).convert("RGB"))
            return PortraitResult(
                avatar_id=avatar_id,
                original_size=tuple(meta.get("original_size", [0, 0])),
                output_size=tuple(meta.get("output_size", [0, 0])),
                output_path=str(out_path),
                thumbnail_path=str(thumb_path),
                image_base64=_to_b64(img_bytes),
                thumbnail_base64=_to_b64(thumb_bytes),
                processing_method=meta.get("processing_method", "pillow-portrait-processing"),
                sha256=meta.get("sha256", ""),
                created_at=meta.get("created_at", ""),
            )
        except Exception:
            return None

    def exists(self, user_id: str, avatar_id: str) -> bool:
        user_dir = self._storage / _safe_id(user_id)
        return (user_dir / f"{_safe_id(avatar_id)}.png").exists()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, data: bytes) -> None:
        if len(data) > self.config.max_input_bytes:
            raise ValueError(f"Image too large: {len(data)//1024}KB > {self.config.max_input_bytes//1024}KB limit")
        if len(data) < 100:
            raise ValueError("Image data too small to be a valid image file")

    def _validate_dimensions(self, img: Image.Image) -> None:
        w, h = img.size
        if w < self.config.min_input_size or h < self.config.min_input_size:
            raise ValueError(f"Image too small: {w}×{h} — minimum {self.config.min_input_size}px per side")

    def _process(self, img: Image.Image) -> Image.Image:
        """Core processing pipeline."""
        # 1. Square-crop from centre
        img = _centre_square_crop(img)

        # 2. Resize to output size
        img = img.resize((self.config.output_size, self.config.output_size), Image.LANCZOS)

        # 3. Mild smoothing to reduce compression artefacts
        img = img.filter(ImageFilter.SMOOTH_MORE)

        # 4. Contrast enhancement
        img = ImageEnhance.Contrast(img).enhance(self.config.enhance_contrast)

        # 5. Colour saturation
        img = ImageEnhance.Color(img).enhance(self.config.enhance_color)

        # 6. Sharpness
        img = ImageEnhance.Sharpness(img).enhance(self.config.enhance_sharpness)

        # 7. Auto-levels (expand histogram to full range)
        img = ImageOps.autocontrast(img, cutoff=0.5)

        return img

    def _to_png_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _centre_square_crop(img: Image.Image) -> Image.Image:
    """Crop to a centred square using the shorter dimension."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _safe_id(s: str) -> str:
    """Strip unsafe characters from an ID for use in filenames."""
    return "".join(c for c in s if c.isalnum() or c in "-_")[:64] or "default"


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _to_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")
