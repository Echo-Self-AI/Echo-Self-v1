#!/usr/bin/env python3
"""
benchmark_watermark.py
======================
Benchmark classical spread-spectrum audio watermarking.

Usage::

    cd /home/raghu/Documents/projects/echoself/ai
    .venv/bin/python scripts/benchmark_watermark.py
"""

from __future__ import annotations

import math
import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from echoself_ai.watermark import ClassicalWatermarkConfig, SpreadSpectrumWatermarker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_tone(
    frequency: float = 440.0,
    duration: float = 2.0,
    sample_rate: int = 16_000,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Generate a mono sinusoidal tone as a 1-D float32 tensor."""
    n_samples = int(duration * sample_rate)
    t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
    return amplitude * torch.sin(2.0 * math.pi * frequency * t)


def add_gaussian_noise_at_snr(signal: torch.Tensor, target_snr_db: float) -> torch.Tensor:
    """Add Gaussian noise to *signal* so that SNR ≈ *target_snr_db* dB."""
    signal_power = torch.mean(signal ** 2).item()
    if signal_power < 1e-20:
        return signal.clone()
    noise_power_target = signal_power / (10.0 ** (target_snr_db / 10.0))
    noise_std = math.sqrt(noise_power_target)
    noise = torch.randn_like(signal) * noise_std
    return signal + noise


def _fmt(val: float, width: int = 10) -> str:
    if math.isinf(val):
        return f"{'inf':>{width}}"
    return f"{val:>{width}.4f}"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    sample_rate = 16_000
    duration = 2.0
    watermark_id = "echoself-benchmark-wm-001"
    strengths = [0.001, 0.01, 0.05]

    print("=" * 72)
    print("EchoSelf — Classical Spread-Spectrum Audio Watermark Benchmark")
    print("=" * 72)
    frequency = 440.0
    print(f"Signal  : {frequency:.0f} Hz sine tone, "
          f"{duration:.1f} s @ {sample_rate} Hz")
    print(f"WM ID   : {watermark_id}")
    print()

    # Generate clean signal
    clean = generate_tone(
        frequency=frequency,
        duration=duration,
        sample_rate=sample_rate,
        amplitude=0.5,
    )

    # -----------------------------------------------------------------------
    # Table 1: clean embedding at three different strengths
    # -----------------------------------------------------------------------
    header = (
        f"{'Strength':>10} | "
        f"{'SNR (dB)':>10} | "
        f"{'Corr':>10} | "
        f"{'Detected':>8} | "
        f"{'BER':>8}"
    )
    sep = "-" * len(header)
    print("Table 1 — Clean signal embedding / detection")
    print(sep)
    print(header)
    print(sep)

    for alpha in strengths:
        config = ClassicalWatermarkConfig(
            watermark_id=watermark_id,
            strength=alpha,
            sample_rate=sample_rate,
        )
        wm = SpreadSpectrumWatermarker(config)

        watermarked = wm.embed(clean, watermark_id)
        snr = wm.compute_snr(clean, watermarked)
        detected, corr, ber = wm.extract(watermarked, watermark_id)

        print(
            f"{alpha:>10.3f} | "
            f"{_fmt(snr)} | "
            f"{_fmt(corr)} | "
            f"{'YES' if detected else 'NO':>8} | "
            f"{ber:>8.4f}"
        )

    print(sep)
    print()

    # -----------------------------------------------------------------------
    # Table 2: robustness — add Gaussian noise (SNR=20 dB) then re-detect
    # -----------------------------------------------------------------------
    noise_snr_db = 20.0
    print(f"Table 2 — Robustness after Gaussian channel noise ({noise_snr_db:.0f} dB SNR)")
    print(sep)
    print(header)
    print(sep)

    for alpha in strengths:
        config = ClassicalWatermarkConfig(
            watermark_id=watermark_id,
            strength=alpha,
            sample_rate=sample_rate,
        )
        wm = SpreadSpectrumWatermarker(config)

        watermarked = wm.embed(clean, watermark_id)
        # Add channel noise on top of the already-watermarked signal
        noisy = add_gaussian_noise_at_snr(watermarked, noise_snr_db)
        noisy = torch.clamp(noisy, -1.0, 1.0)

        snr = wm.compute_snr(clean, watermarked)   # intrinsic wm SNR
        detected, corr, ber = wm.extract(noisy, watermark_id)

        print(
            f"{alpha:>10.3f} | "
            f"{_fmt(snr)} | "
            f"{_fmt(corr)} | "
            f"{'YES' if detected else 'NO':>8} | "
            f"{ber:>8.4f}"
        )

    print(sep)
    print()

    # -----------------------------------------------------------------------
    # False-positive check: detect on a signal that was NOT watermarked
    # -----------------------------------------------------------------------
    print("False-positive check (no watermark embedded)")
    config = ClassicalWatermarkConfig(
        watermark_id=watermark_id,
        strength=0.01,
        sample_rate=sample_rate,
    )
    wm = SpreadSpectrumWatermarker(config)
    detected_fp, corr_fp, ber_fp = wm.extract(clean, watermark_id)
    print(f"  Detected: {'YES' if detected_fp else 'NO'}  |  Corr: {corr_fp:.6f}  |  BER: {ber_fp:.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
