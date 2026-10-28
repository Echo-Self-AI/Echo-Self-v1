"""Unit tests for validated audio ingestion, format validation, and resampling."""
from io import BytesIO
from pathlib import Path
import struct
import wave

import pytest
import torch

from echoself_ai.voice.audio import (
    AudioConfig,
    AudioDurationError,
    AudioFileSizeError,
    CorruptAudioError,
    InvalidAudioFormatError,
    generate_synthetic_tone,
    ingest_audio,
    resample_waveform,
    write_wav,
)


def test_ingest_valid_wav_file(tmp_path: Path):
    audio_path = tmp_path / "valid.wav"
    tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
    write_wav(audio_path, tone, sample_rate=16000)

    wf, sr = ingest_audio(audio_path, target_sr=16000)
    assert sr == 16000
    assert wf.dim() == 1
    assert len(wf) == 8000
    assert torch.isfinite(wf).all()
    assert torch.max(torch.abs(wf)) <= 1.0


def test_ingest_bytes_and_bytesio():
    tone = generate_synthetic_tone(freq_hz=220.0, duration_seconds=0.3, sample_rate=16000)
    buf = BytesIO()
    write_wav(buf, tone, sample_rate=16000)
    raw_bytes = buf.getvalue()

    # From bytes
    wf1, sr1 = ingest_audio(raw_bytes, target_sr=16000)
    assert sr1 == 16000
    assert len(wf1) == len(tone)

    # From BytesIO
    buf2 = BytesIO(raw_bytes)
    wf2, sr2 = ingest_audio(buf2, target_sr=16000)
    assert sr2 == 16000
    assert torch.allclose(wf1, wf2)


def test_resample_waveform_interpolation():
    orig_sr = 8000
    target_sr = 16000
    tone_8k = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=orig_sr)
    assert len(tone_8k) == 4000

    tone_16k = resample_waveform(tone_8k, orig_sr=orig_sr, target_sr=target_sr)
    assert len(tone_16k) == 8000
    assert torch.isfinite(tone_16k).all()


def test_ingest_stereo_to_mono(tmp_path: Path):
    stereo_path = tmp_path / "stereo.wav"
    sample_rate = 16000
    n_frames = 4000  # 0.25s
    # Write 2-channel 16-bit PCM WAV
    with wave.open(str(stereo_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        ints = [1000, 2000] * n_frames
        raw_bytes = struct.pack(f"<{len(ints)}h", *ints)
        wf.writeframes(raw_bytes)

    mono, sr = ingest_audio(stereo_path, target_sr=16000)
    assert sr == 16000
    assert mono.dim() == 1
    assert len(mono) == n_frames
    # Mean of 1000 and 2000 is 1500 / 32768
    expected = 1500.0 / 32768.0
    assert torch.isclose(mono[0], torch.tensor(expected), atol=1e-3)



def test_reject_non_wav_data(tmp_path: Path):
    corrupt_file = tmp_path / "not_audio.wav"
    corrupt_file.write_text("This is plain text pretending to be WAV audio.")

    with pytest.raises(InvalidAudioFormatError):
        ingest_audio(corrupt_file)


def test_reject_truncated_header(tmp_path: Path):
    trunc_file = tmp_path / "trunc.wav"
    trunc_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    with pytest.raises(CorruptAudioError):
        ingest_audio(trunc_file)


def test_reject_duration_out_of_bounds(tmp_path: Path):
    # Too short (< 0.1s)
    short_path = tmp_path / "too_short.wav"
    short_tone = generate_synthetic_tone(duration_seconds=0.05, sample_rate=16000)
    write_wav(short_path, short_tone, sample_rate=16000)

    cfg = AudioConfig(min_duration_seconds=0.1, max_duration_seconds=5.0)
    with pytest.raises(AudioDurationError):
        ingest_audio(short_path, config=cfg)

    # Too long (> 5.0s)
    long_path = tmp_path / "too_long.wav"
    long_tone = generate_synthetic_tone(duration_seconds=5.5, sample_rate=16000)
    write_wav(long_path, long_tone, sample_rate=16000)

    with pytest.raises(AudioDurationError):
        ingest_audio(long_path, config=cfg)


def test_reject_excessive_file_size(tmp_path: Path):
    tone = generate_synthetic_tone(duration_seconds=0.5, sample_rate=16000)
    buf = BytesIO()
    write_wav(buf, tone, sample_rate=16000)
    raw = buf.getvalue()

    cfg = AudioConfig(max_file_size_bytes=100)  # Unrealistic small limit
    with pytest.raises(AudioFileSizeError):
        ingest_audio(raw, config=cfg)
