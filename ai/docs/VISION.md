# EchoSelf Visual AI and Avatar Subsystem

## Overview

The EchoSelf Visual AI subsystem provides local, privacy-first computer vision, visual feature representation, and neural avatar generation without relying on paid APIs or external cloud services.

```text
ai/src/echoself_ai/vision/
├── __init__.py
├── image_ops.py       # Image loading, saving, normalization, and synthetic test image generation
├── encoder.py         # Multi-scale residual visual encoder with L2-normalized embeddings
├── avatar.py          # Avatar profile, neural generator, identity loss, and consent gating
└── metrics.py         # Image quality metrics (PSNR, SSIM, MSE) and generation benchmark pipeline
```

---

## Core Modules

### 1. Image Operations & Normalization (`image_ops.py`)
- Standardizes all visual representations to normalized PyTorch float tensors in the $[-1.0, 1.0]$ interval with channel-first order `(C, H, W)`.
- Implements aspect-ratio-preserving square resizing with zero-padding.
- Provides test image generators (solid colors, geometric gradients, synthetic identity patterns) enabling end-to-end verification without downloading external datasets.

### 2. Visual Feature Extractor (`encoder.py`)
- `VisualEncoder`: A deep convolutional network with residual connections (`ResConvBlock`), batch normalization, GELU activations, and adaptive average pooling.
- Projects arbitrary-sized RGB images down to fixed-dimension $d_{\text{emb}}$ (default 128) L2-normalized vectors:
  $$\|z\|_{2} = 1.0$$
- Facilitates identity similarity comparisons via standard cosine similarity:
  $$\text{sim}(z_1, z_2) = \frac{z_1 \cdot z_2}{\|z_1\|_2 \|z_2\|_2}$$

### 3. Avatar Identity Profile & Generator (`avatar.py`)
- `AvatarProfile`: Represents an avatar identity with canonical visual embedding, display attributes, and JSON serialization.
- `AvatarGenerator`: A convolutional transposed decoder synthesizing $64 \times 64$ avatar images conditioned on concatenated identity embeddings $z_{\text{id}}$ and expression/pose conditioning vectors $c$:
  $$\hat{x} = G(z_{\text{id}}, c) \in [-1.0, 1.0]^{3 \times 64 \times 64}$$
- `compute_identity_consistency_loss`: Enforces facial and visual consistency during generation training:
  $$\mathcal{L}_{\text{id}} = 1.0 - \cos\left(E(\hat{x}), z_{\text{id}}\right)$$

### 4. Cryptographic Visual Consent & Privacy (`avatar.py`)
- `VisualConsentManager`: Enforces explicit user authorization before generating or animating any avatar.
- **Verification Hash**: SHA-256 cryptographic hash over user ID, avatar ID, scope, timestamp, and authorization statement.
- **Strict Gating**: Calling `AvatarGenerator.generate()` with unconsented or revoked avatars raises `UnauthorizedAvatarError`.
- **GDPR Compliance (Right to Erasure)**: `purge_avatar_data()` permanently deletes consent records, profile weights, and visual artifacts.

### 5. Quality Metrics & Evaluation (`metrics.py`)
- `compute_mse`: Mean squared error between predicted and target image tensors.
- `compute_psnr`: Peak Signal-to-Noise Ratio in decibels (dB), calibrated for $[-1.0, 1.0]$ dynamic range ($2.0$).
- `compute_ssim`: Multi-channel Structural Similarity Index using pure PyTorch tensor convolutions.
- `evaluate_avatar_generation`: Measures inference throughput (FPS), per-frame latency (ms), and identity preservation similarity.

---

## Performance & Hardware Benchmarks

Evaluated on local workstation (12-core CPU, NVIDIA RTX 3050 Laptop GPU):
- **CPU Generation Latency**: ~10–18 ms per $64 \times 64$ avatar frame (~55–90 FPS).
- **GPU Generation Latency**: < 2 ms per frame (> 500 FPS).
- **RAM Footprint**: Under 180 MB.
- **Identity Embedding Extraction**: < 5 ms per image on CPU.

---

## Technical Limitations & Roadmap

- **From-Scratch Convolutional Baseline**: EchoSelf's native visual generator is a lightweight convolutional neural model designed for local, zero-cost real-time inference. It produces stylized facial frames rather than photorealistic multi-billion parameter diffusion renders.
- **Resolution**: Default native resolution is $64 \times 64$ (extensible to $128 \times 128$). Higher resolutions will be supported via modular super-resolution upsamplers in subsequent phases.
