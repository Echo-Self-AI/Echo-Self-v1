from pathlib import Path
import pytest
import torch

from echoself_ai.voice.audio import (
    AudioConfig,
    generate_synthetic_tone,
    load_wav,
    normalize_waveform,
    write_wav,
)


def test_audio_config_validation():
    with pytest.raises(ValueError, match="sample_rate"):
        AudioConfig(sample_rate=4000)

    with pytest.raises(ValueError, match="min_duration_seconds"):
        AudioConfig(min_duration_seconds=0.0)

    with pytest.raises(ValueError, match="max_duration_seconds"):
        AudioConfig(min_duration_seconds=5.0, max_duration_seconds=4.0)


def test_synthetic_tone_generation():
    tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)
    assert tone.shape == (8000,)
    assert tone.dtype == torch.float32
    assert torch.max(torch.abs(tone)) <= 0.81


def test_wav_write_and_load_round_trip(tmp_path: Path):
    audio_file = tmp_path / "test.wav"
    original = generate_synthetic_tone(freq_hz=220.0, duration_seconds=0.25, sample_rate=16000)

    write_wav(audio_file, original, sample_rate=16000)
    assert audio_file.exists()

    loaded, sr = load_wav(audio_file)
    assert sr == 16000
    assert loaded.shape == original.shape

    # 16-bit PCM quantization has precision within ~1/32768 (~3e-5)
    torch.testing.assert_close(loaded, original, rtol=1e-3, atol=1e-3)


def test_load_wav_duration_constraints(tmp_path: Path):
    audio_file = tmp_path / "short.wav"
    # 0.05 seconds audio
    short_tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.05, sample_rate=16000)
    write_wav(audio_file, short_tone, sample_rate=16000)

    config = AudioConfig(min_duration_seconds=0.1, max_duration_seconds=1.0)
    with pytest.raises(ValueError, match="less than minimum"):
        load_wav(audio_file, config=config)


def test_normalize_waveform():
    unnorm = torch.tensor([0.1, -0.5, 0.2])
    norm = normalize_waveform(unnorm, target_peak=0.95)
    assert torch.isclose(torch.max(torch.abs(norm)), torch.tensor(0.95))


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_wav("does_not_exist_12345.wav")
