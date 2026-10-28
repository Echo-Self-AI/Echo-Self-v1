from pathlib import Path
from PIL import Image
import pytest
import torch

from echoself_ai.vision.image_ops import (
    ImageConfig,
    generate_synthetic_image,
    load_image,
    save_image,
    tensor_to_pil,
)


def test_image_config_defaults():
    cfg = ImageConfig()
    assert cfg.target_size == (128, 128)
    assert cfg.max_file_size_bytes > 0
    assert cfg.normalize_range == (-1.0, 1.0)


def test_generate_synthetic_image_patterns():
    checker = generate_synthetic_image(width=64, height=64, pattern="checkerboard")
    assert checker.shape == (3, 64, 64)
    assert checker.min() >= -1.0
    assert checker.max() <= 1.0

    gradient = generate_synthetic_image(width=32, height=32, pattern="gradient")
    assert gradient.shape == (3, 32, 32)
    assert gradient.min() >= -1.0
    assert gradient.max() <= 1.0


def test_image_save_and_load_round_trip(tmp_path: Path):
    img_path = tmp_path / "sample.png"
    synth_tensor = generate_synthetic_image(width=64, height=64, pattern="checkerboard")

    save_image(img_path, synth_tensor)
    assert img_path.exists()

    cfg = ImageConfig(target_size=(64, 64))
    loaded = load_image(img_path, config=cfg)

    assert loaded.shape == (3, 64, 64)
    # Color values after 8-bit quantization are close within ~1/128 (~0.015)
    torch.testing.assert_close(loaded, synth_tensor, atol=0.03, rtol=0.03)


def test_load_image_resizing(tmp_path: Path):
    img_path = tmp_path / "original.png"
    # Create 32x32 image
    img = Image.new("RGB", (32, 32), color="blue")
    img.save(img_path)

    # Load and resize to 64x64
    cfg = ImageConfig(target_size=(64, 64))
    loaded = load_image(img_path, config=cfg)
    assert loaded.shape == (3, 64, 64)


def test_load_image_file_size_limit(tmp_path: Path):
    img_path = tmp_path / "oversized.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(img_path)

    # Restrict max size to 10 bytes
    cfg = ImageConfig(max_file_size_bytes=10)
    with pytest.raises(ValueError, match="exceeds maximum allowable limit"):
        load_image(img_path, config=cfg)


def test_load_nonexistent_image_raises():
    with pytest.raises(FileNotFoundError):
        load_image("does_not_exist_image_xyz.png")


def test_tensor_to_pil_invalid_shape():
    invalid_tensor = torch.zeros(2, 64, 64)
    with pytest.raises(ValueError, match="Expected tensor of shape"):
        tensor_to_pil(invalid_tensor)
