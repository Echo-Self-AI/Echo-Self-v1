# EchoSelf Voice AI Subsystem

## Overview

The EchoSelf Voice AI subsystem provides zero-cost, local speech processing and synthesis without requiring paid external APIs or proprietary cloud services.

```text
ai/src/echoself_ai/voice/
├── __init__.py
├── audio.py           # Zero-dependency 16-bit PCM WAV loader, parser, and writer
├── features.py        # STFT and 80-channel Log Mel-Spectrogram feature extractor
├── consent.py         # Cryptographic user authorization and consent gating manager
├── acoustic_model.py  # 1D Convolutional / residual text-to-mel acoustic predictor
└── vocoder.py         # Griffin-Lim phase vocoder and end-to-end VoiceSynthesizer
```

## Core Modules

### 1. Zero-Dependency WAV Processing (`audio.py`)
- Reads and writes standard 16-bit PCM WAV files natively via Python standard library (`wave`, `struct`) and PyTorch tensors.
- Multichannel audio is automatically mixed down to single-channel mono.
- Enforces duration boundaries and peak amplitude normalization (`normalize_waveform`).

### 2. Mel Feature Extraction (`features.py`)
- Standard 80-channel log-mel spectrogram extraction:
  - Sample rate: 16,000 Hz.
  - FFT size: 512 bins (32 ms window).
  - Hop size: 128 samples (8 ms hop interval).
  - Mel frequency range: 0 Hz to 8,000 Hz.

### 3. Voice Consent & Privacy Gating (`consent.py`)
- **Strict Gating**: No voice synthesis, speaker conditioning, or voice cloning can execute without an active, non-revoked `VoiceConsentRecord`.
- **Right to Erasure (GDPR)**: Calling `purge_speaker_data()` purges all speaker recordings, embeddings, and caches from storage upon consent revocation.

### 4. Acoustic Model & Griffin-Lim Vocoder (`acoustic_model.py`, `vocoder.py`)
- `AcousticModel`: 1D residual convolutional neural network mapping token IDs and optional authorized speaker IDs to 80-channel mel-spectrograms.
- `GriffinLimVocoder`: Mathematical iterative phase reconstruction inverting mel-spectrograms to audible waveforms without needing large pretrained vocoder checkpoints.
- `VoiceSynthesizer`: End-to-end synthesizer taking tokens and generating standard `.wav` audio.

## Latency & Hardware Profile
- CPU execution: Real-time factor (RTF) < 0.25 on 12-core CPU (synthesizes 1 second of speech in ~250ms).
- RAM footprint: Under 150 MB.
