# EchoSelf AI Layer — Media Interfaces & Ingestion Specification

This document specifies the data contracts, validation rules, error taxonomy, and tensor representations for local audio and visual media ingestion in EchoSelf.

---

## 1. Interface Audit & Current Limitations

Prior to Phase 1, the AI service accepted media requests via `ProcessRequest` payloads, but used simplified mock behavior for embeddings:
- `_handle_voice_embedding`: Computed a pseudo-embedding by hashing strings with SHA-512 rather than analyzing speech acoustics.
- `_handle_avatar_embedding`: Fed Gaussian noise `torch.randn(1, 3, 64, 64)` into `VisualEncoder` rather than processing a user's portrait image.

Phase 1 establishes real media ingestion pipelines for audio (`.wav`) and images (`.png`, `.jpg`, `.jpeg`, `.webp`), ensuring valid data is ingested, verified against biometric consent, and processed by actual PyTorch encoders.

---

## 2. Audio Ingestion Specification (`echoself_ai.voice.audio`)

### Contract & Requirements

| Property | Requirement | Validation Rule / Behavior |
|---|---|---|
| **Container Format** | RIFF WAV | Rejects non-WAV headers with `InvalidAudioError`. |
| **Encoding** | 16-bit Linear PCM | Rejects compressed/floating-point WAVs without 16-bit PCM header. |
| **Channels** | Mono or Stereo | Stereo channels are automatically downmixed to mono via mean channel reduction. |
| **Sample Rate** | 8 kHz to 48 kHz | Standardized to 16,000 Hz using linear interpolation resampling. |
| **Duration Bounds** | 0.2s minimum, 60s maximum | Files outside range rejected with `AudioDurationError`. |
| **File Size Limit** | Maximum 25 MB | Checked before memory allocation to prevent resource exhaustion. |
| **Data Integrity** | Finite values only | Checks for `NaN` and `Inf` float conversions; rejects corrupt frames. |
| **Output Tensor** | `(num_samples,)` float32 | Amplitude normalized in range `[-1.0, 1.0]`. |

### Acoustic Feature Extraction Pipeline

$$\text{WAV File / Base64} \xrightarrow{\text{validate}} \text{Mono } 16\text{kHz Waveform } [-1.0, 1.0] \xrightarrow{\text{STFT + Filterbank}} 80\text{-channel Log-Mel Spectrogram } [80, T]$$

Acoustic identity vectors (64-d) are extracted by computing mean and standard deviation pooling across the 80 mel channels:
$$\mathbf{e}_{\text{voice}} = \text{L2\_Normalize}\left(\left[\mu_1, \dots, \mu_{32}, \sigma_1, \dots, \sigma_{32}\right]\right) \in \mathbb{R}^{64}$$

---

## 3. Image Ingestion Specification (`echoself_ai.vision.image_ops`)

### Contract & Requirements

| Property | Requirement | Validation Rule / Behavior |
|---|---|---|
| **Supported Formats** | PNG, JPEG, WebP | Verified via magic bytes and PIL decoder. Rejects unsupported formats. |
| **File Size Limit** | Maximum 15 MB | Checked prior to full image decompression. |
| **Dimension Bounds** | Min $32 \times 32$, Max $4096 \times 4096$ | Rejects degenerate zero-size or extremely large images. |
| **Color Space** | RGB (3 channels) | Grayscale images are converted to 3-channel RGB; alpha channels are discarded safely. |
| **Standardized Size** | $64 \times 64$ for avatars | Bilinear interpolation downsampling to model resolution. |
| **Output Tensor** | `(3, 64, 64)` float32 | Scaled to normalized range `[-1.0, 1.0]`. |

### Visual Identity Extraction Pipeline

$$\text{Image File / Base64} \xrightarrow{\text{validate}} \text{RGB Tensor } (3, 64, 64) \in [-1.0, 1.0] \xrightarrow{\text{VisualEncoder}} \text{L2-Normalized Identity Vector } \in \mathbb{R}^{128}$$

---

## 4. Error Taxonomy & Rejection Protocol

All media validation errors inherit from `MediaValidationError` to enable deterministic error classification:

```
MediaValidationError (Exception)
├── AudioValidationError
│   ├── InvalidAudioFormatError (Non-WAV or unsupported PCM bit depth)
│   ├── AudioDurationError (Duration < 0.2s or > 60s)
│   ├── AudioFileSizeError (Payload exceeds size limit)
│   └── CorruptAudioError (Truncated file, NaN or Inf samples)
└── ImageValidationError
    ├── InvalidImageFormatError (Unrecognized image container)
    ├── ImageDimensionError (Image dimensions smaller than 32x32 or corrupt)
    ├── ImageFileSizeError (Payload exceeds 15 MB limit)
    └── CorruptImageError (PIL decoding exception)
```

When an ingestion error occurs during `EchoSelfAIService.process`:
1. The error is intercepted.
2. The job fails with `status="failed"`.
3. The specific error message and validation error type are returned in `metadata.error_type`.
4. No synthetic fallback or dummy tensor is silently substituted.
