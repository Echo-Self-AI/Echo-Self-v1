import pytest
import torch

from echoself_ai.voice.audio import generate_synthetic_tone
from echoself_ai.voice.features import LogMelSpectrogram, create_mel_filterbank


def test_mel_filterbank_shape_and_bounds():
    fb = create_mel_filterbank(sample_rate=16000, n_fft=512, n_mels=80)
    assert fb.shape == (80, 257)
    assert fb.min() >= 0.0
    assert fb.max() <= 1.0


def test_log_mel_spectrogram_output_shape():
    extractor = LogMelSpectrogram(sample_rate=16000, n_fft=512, hop_length=128, n_mels=80)
    # 0.5 seconds = 8000 samples
    waveform = generate_synthetic_tone(freq_hz=440.0, duration_seconds=0.5, sample_rate=16000)

    mel = extractor(waveform)
    assert mel.dim() == 3
    assert mel.size(0) == 1  # batch
    assert mel.size(1) == 80  # n_mels
    # num_frames approx 8000 / 128 + 1
    assert mel.size(2) > 50
    assert not torch.isnan(mel).any()
    assert not torch.isinf(mel).any()


def test_log_mel_frequency_discrimination():
    """Verify that a high frequency tone concentrates energy in higher mel bands than a low tone."""
    extractor = LogMelSpectrogram(sample_rate=16000, n_fft=512, hop_length=128, n_mels=80)

    low_tone = generate_synthetic_tone(freq_hz=300.0, duration_seconds=0.25, sample_rate=16000)
    high_tone = generate_synthetic_tone(freq_hz=3000.0, duration_seconds=0.25, sample_rate=16000)

    mel_low = extractor(low_tone).squeeze(0).mean(dim=-1)  # shape (80,)
    mel_high = extractor(high_tone).squeeze(0).mean(dim=-1)

    peak_low_channel = torch.argmax(mel_low).item()
    peak_high_channel = torch.argmax(mel_high).item()

    assert peak_high_channel > peak_low_channel


def test_batched_waveform_input():
    extractor = LogMelSpectrogram(sample_rate=16000, n_fft=512, hop_length=128, n_mels=80)
    batch = torch.randn(4, 4000)
    mel = extractor(batch)

    assert mel.shape == (4, 80, 32)
