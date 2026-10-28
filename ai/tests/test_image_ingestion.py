"""Unit tests for validated image ingestion, format validation, and dimension enforcement."""
from io import BytesIO
from pathlib import Path
from PIL import Image
import pytest
import torch

from echoself_ai.vision.image_ops import (
    CorruptImageError,
    ImageConfig,
    ImageDimensionError,
    ImageFileSizeError,
    InvalidImageFormatError,
    generate_synthetic_image,
    ingest_image,
    save_image,
    tensor_to_pil,
)


def test_ingest_valid_png_and_jpeg(tmp_path: Path):
    syn_tensor = generate_synthetic_image(width=128, height=128, pattern="checkerboard")

    png_path = tmp_path / "test.png"
    save_image(png_path, syn_tensor)

    out_png = ingest_image(png_path, target_size=(64, 64))
    assert out_png.shape == (3, 64, 64)
    assert torch.isfinite(out_png).all()
    assert -1.0 <= out_png.min() and out_png.max() <= 1.0

    # JPEG test
    jpg_path = tmp_path / "test.jpg"
    pil_img = tensor_to_pil(syn_tensor)
    pil_img.save(jpg_path, format="JPEG")

    out_jpg = ingest_image(jpg_path, target_size=(64, 64))
    assert out_jpg.shape == (3, 64, 64)


def test_ingest_bytes_and_bytesio():
    syn_tensor = generate_synthetic_image(width=64, height=64, pattern="gradient")
    pil_img = tensor_to_pil(syn_tensor)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Ingest from raw bytes
    tensor_from_bytes = ingest_image(raw_bytes, target_size=(32, 32))
    assert tensor_from_bytes.shape == (3, 32, 32)

    # Ingest from BytesIO
    buf2 = BytesIO(raw_bytes)
    tensor_from_bio = ingest_image(buf2, target_size=(32, 32))
    assert tensor_from_bio.shape == (3, 32, 32)
    assert torch.allclose(tensor_from_bytes, tensor_from_bio)


def test_reject_unsupported_format(tmp_path: Path):
    bmp_path = tmp_path / "test.bmp"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(bmp_path, format="BMP")

    with pytest.raises(InvalidImageFormatError):
        ingest_image(bmp_path)


def test_reject_dimension_out_of_bounds():
    # Too small (< 32x32)
    tiny_img = Image.new("RGB", (16, 16), color="blue")
    buf = BytesIO()
    tiny_img.save(buf, format="PNG")

    with pytest.raises(ImageDimensionError):
        ingest_image(buf.getvalue())

    # Too large (> 4096)
    cfg = ImageConfig(max_dimension=100)
    img_150 = Image.new("RGB", (150, 150), color="green")
    buf2 = BytesIO()
    img_150.save(buf2, format="PNG")

    with pytest.raises(ImageDimensionError):
        ingest_image(buf2.getvalue(), config=cfg)


def test_reject_empty_and_corrupt_images(tmp_path: Path):
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(CorruptImageError):
        ingest_image(empty_file)

    corrupt_file = tmp_path / "garbage.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\nCorruptPayloadNotRealPngData")

    with pytest.raises(CorruptImageError):
        ingest_image(corrupt_file)


def test_reject_excessive_file_size():
    img = Image.new("RGB", (64, 64), color="yellow")
    buf = BytesIO()
    img.save(buf, format="PNG")

    cfg = ImageConfig(max_file_size_bytes=50)  # Unrealistic small limit
    with pytest.raises(ImageFileSizeError):
        ingest_image(buf.getvalue(), config=cfg)
