# EchoSelf Prompt-to-Video Subsystem

## Overview

The EchoSelf Prompt-to-Video subsystem provides local, privacy-first video sequence generation, animated avatar synthesis, and temporal consistency modeling without reliance on external cloud APIs or paid third-party platforms.

```text
ai/src/echoself_ai/video/
├── __init__.py
├── frames.py      # VideoConfig, FrameSequence container, animated GIF & PNG directory I/O
├── temporal.py    # Temporal consistency loss, (2D+1D) spatiotemporal conv, neural frame interpolator
├── pipeline.py    # PromptToVideoGenerator, prompt encoder, identity conditioning & consent gating
└── metrics.py     # Video PSNR, Video SSIM, temporal flicker variance & pipeline benchmark
```

---

## Core Modules & Mathematical Principles

### 1. Temporal Container & Sequence I/O (`frames.py`)
- `FrameSequence`: Standardized 4D PyTorch tensor container of shape $(T, C, H, W)$ strictly normalized in the $[-1.0, 1.0]$ range.
- Native export and import:
  - Animated GIF format via Pillow (`save_gif`, `load_gif`) preserving target duration per frame.
  - Sequenced PNG image directories (`save_frame_directory`, `load_frame_directory`) with frame rate metadata.
- Deterministic synthetic video generator (`generate_synthetic_video`) supporting standard procedural patterns ("bounce", "sweep") for end-to-end verification without external assets.

### 2. Temporal Consistency & Dynamics (`temporal.py`)
- **Temporal Consistency Loss**: Penalizes frame-to-frame pixel flickering and second-order motion acceleration (jerk):
  $$\mathcal{L}_{\text{temporal}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \|x_{t} - x_{t-1}\|_{1} + \lambda_{\text{smooth}} \frac{1}{T-2} \sum_{t=1}^{T-2} \|(x_{t+1} - x_t) - (x_t - x_{t-1})\|_2^2$$
- **Separable (2D + 1D) Spatiotemporal Convolution (`TemporalConvBlock`)**:
  Decomposes 3D spatio-temporal convolution into spatial 2D conv across spatial planes $(1 \times K \times K)$ followed by temporal 1D conv $(K \times 1 \times 1)$ across the sequence dimension with residual skip connections. Eliminates compute bottleneck of 3D convolutions.
- **Neural Frame Interpolator (`FrameInterpolator`)**:
  Predicts smooth in-between frames $x_\alpha$ given anchor frames $x_A, x_B$ and temporal interpolation coordinate $\alpha \in [0.0, 1.0]$. Supports arbitrary slow-motion or temporal upsampling (`interpolate_sequence`).

### 3. Prompt-to-Video Conditioning Pipeline (`pipeline.py`)
- `PromptTextEncoder`: Encodes natural language strings into continuous semantic condition vectors $z_{\text{text}} \in \mathbb{R}^{D_{\text{prompt}}}$ via byte embedding and 1D residual convolutions.
- `PromptToVideoGenerator`:
  - Fuses text prompt conditions with canonical avatar identity embeddings $z_{\text{id}} \in \mathbb{R}^{128}$ from `AvatarProfile` / `VisualEncoder`.
  - Harmonically modulates temporal latents with multi-frequency sinusoidal positional time embeddings:
    $$\phi_k(t) = \left[\sin\left(2\pi \frac{t}{T-1} \cdot k\right), \cos\left(2\pi \frac{t}{T-1} \cdot k\right)\right]$$
  - Progressively decodes spatial features using transposed convolutional stages.
  - Refines multi-frame coherence via spatiotemporal smoothing.
- **Strict Privacy Consent Gating**:
  - Generation targeting a specific user avatar requires verified, non-revoked authorization via `VisualConsentManager`.
  - Unconsented generation raises `UnauthorizedAvatarError`.

### 4. Video Quality Metrics & Evaluation (`metrics.py`)
- `compute_video_psnr`: Computes average frame-wise Peak Signal-to-Noise Ratio (PSNR) across matching sequences in decibels (dB).
- `compute_video_ssim`: Computes average Structural Similarity Index (SSIM) across video frames.
- `compute_temporal_flicker_score`: Quantifies the variance of sequential frame differences:
  $$\text{FlickerScore} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T-1} \left(\|x_t - x_{t-1}\|_1 - \mu_{\Delta}\right)^2}$$
- `evaluate_video_pipeline`: Measures video generation latency (ms/video, ms/frame), FPS throughput, temporal loss, and flicker score.

---

## Hardware Benchmarks & Performance Profile

Evaluated on local workstation (12 CPU cores, NVIDIA RTX 3050 Laptop GPU):
- **CPU Generation Throughput**: ~40–60 FPS for $64 \times 64$ frames (~250–350 ms for a 16-frame animated GIF).
- **GPU Generation Throughput**: > 250 FPS (< 60 ms for a 16-frame animated sequence).
- **RAM Footprint**: Under 250 MB peak memory during video inference.
- **GIF File Size**: ~15–35 KB for a 16-frame $64 \times 64$ animated clip.

---

## Technical Constraints & Honest Baseline Comparison

- **From-Scratch Convolutional/Spatiotemporal Architecture**: EchoSelf implements a lightweight native spatiotemporal neural network tailored for real-time local execution without multi-gigabyte foundation model weights.
- **Visual Style**: Renders clean, stylized avatar animation sequences rather than hyper-photorealistic multi-minute cinematic video.
- **Frame Resolution**: Default $64 \times 64$ native resolution (extensible to $128 \times 128$) optimized for responsive local avatar interactions.
