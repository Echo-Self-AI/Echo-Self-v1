"""Integration tests for real media ingestion in EchoSelfAIService."""
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

import pytest
import torch

from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest
from echoself_ai.vision.image_ops import generate_synthetic_image, save_image
from echoself_ai.voice.audio import generate_synthetic_tone, write_wav


def test_voice_embedding_with_real_audio_file(tmp_path: Path):
    service = EchoSelfAIService()
    wav_path = tmp_path / "speaker_sample.wav"
    tone = generate_synthetic_tone(freq_hz=300.0, duration_seconds=0.5, sample_rate=16000)
    write_wav(wav_path, tone, sample_rate=16000)

    req = ProcessRequest(
        job_id="test-voice-real-1",
        job_type="voice_embedding",
        user_id="user-voice-real",
        payload={
            "speaker_id": "spk-real",
            "audio_path": str(wav_path),
        },
    )

    result = service.process(req)
    assert result.status == "success"
    assert result.output["speaker_id"] == "spk-real"
    assert result.output["embedding_dim"] == 64
    emb = result.output["embedding"]
    assert len(emb) == 64
    # Check L2 normalization
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_voice_embedding_with_audio_base64():
    service = EchoSelfAIService()
    tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.4, sample_rate=16000)
    buf = BytesIO()
    write_wav(buf, tone, sample_rate=16000)
    b64_audio = base64.b64encode(buf.getvalue()).decode("ascii")

    req = ProcessRequest(
        job_id="test-voice-b64-1",
        job_type="voice_embedding",
        user_id="user-voice-b64",
        payload={
            "speaker_id": "spk-b64",
            "audio_base64": b64_audio,
        },
    )

    result = service.process(req)
    assert result.status == "success"
    assert result.output["embedding_dim"] == 64


def test_voice_embedding_rejects_corrupted_audio(tmp_path: Path):
    service = EchoSelfAIService()
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"RIFF\x00\x00NotRealWavData")

    req = ProcessRequest(
        job_id="test-voice-corrupt-1",
        job_type="voice_embedding",
        user_id="user-corrupt",
        payload={
            "speaker_id": "spk-bad",
            "audio_path": str(corrupt_path),
        },
    )

    result = service.process(req)
    assert result.status == "failed"
    assert "CorruptAudioError" in result.metadata.get("error_type", "") or "Failed" in (result.error or "")


def test_avatar_embedding_with_real_image_file(tmp_path: Path):
    service = EchoSelfAIService()
    img_path = tmp_path / "portrait.png"
    syn_tensor = generate_synthetic_image(width=128, height=128, pattern="checkerboard")
    save_image(img_path, syn_tensor)

    req = ProcessRequest(
        job_id="test-avatar-real-1",
        job_type="avatar_embedding",
        user_id="user-avatar-real",
        payload={
            "avatar_id": "avatar-real-1",
            "image_path": str(img_path),
        },
    )

    result = service.process(req)
    assert result.status == "success"
    assert result.output["avatar_id"] == "avatar-real-1"
    assert result.output["embedding_dim"] == 128
    emb = result.output["embedding"]
    norm = sum(x * x for x in emb) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_avatar_embedding_with_image_base64():
    service = EchoSelfAIService()
    img = Image.new("RGB", (64, 64), color="purple")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

    req = ProcessRequest(
        job_id="test-avatar-b64-1",
        job_type="avatar_embedding",
        user_id="user-avatar-b64",
        payload={
            "avatar_id": "avatar-b64",
            "image_base64": b64_img,
        },
    )

    result = service.process(req)
    assert result.status == "success"
    assert result.output["embedding_dim"] == 128


def test_avatar_embedding_rejects_corrupted_image(tmp_path: Path):
    service = EchoSelfAIService()
    corrupt_img = tmp_path / "bad.png"
    corrupt_img.write_bytes(b"\x89PNG\r\n\x1a\nNotAValidPngFile")

    req = ProcessRequest(
        job_id="test-avatar-bad-1",
        job_type="avatar_embedding",
        user_id="user-avatar-bad",
        payload={
            "avatar_id": "avatar-bad",
            "image_path": str(corrupt_img),
        },
    )

    result = service.process(req)
    assert result.status == "failed"
    assert "CorruptImageError" in result.metadata.get("error_type", "") or "Failed" in (result.error or "")
