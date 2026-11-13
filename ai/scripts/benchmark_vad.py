#!/usr/bin/env python3
"""VAD Benchmark Script — EchoSelf Week 2.

Generates synthetic test audio (tone, silence, mixed), runs both VAD backends,
measures wall-clock latency, and reports the real-time factor (RTF).

RTF = processing_time / audio_duration.  RTF < 1.0 means faster-than-real-time.

Usage::

    cd /home/raghu/Documents/projects/echoself/ai
    .venv/bin/python scripts/benchmark_vad.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sure the package is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from echoself_ai.voice.audio import generate_synthetic_tone
from echoself_ai.voice.vad import create_vad
from echoself_ai.voice.vad_config import VADConfig

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
SILENCE_AMP = 1e-5


def _make_tone(duration_sec: float, amp: float = 0.8) -> torch.Tensor:
    return generate_synthetic_tone(
        freq_hz=440.0,
        duration_seconds=duration_sec,
        sample_rate=SAMPLE_RATE,
        amplitude=amp,
    )


def _make_silence(duration_sec: float) -> torch.Tensor:
    n = int(duration_sec * SAMPLE_RATE)
    return torch.randn(n) * SILENCE_AMP


def _make_mixed(speech_sec: float, silence_sec: float) -> torch.Tensor:
    tone = _make_tone(speech_sec)
    silence = _make_silence(silence_sec)
    return torch.cat([tone, silence, tone])


def _benchmark(
    name: str,
    vad,
    waveform: torch.Tensor,
    audio_duration_sec: float,
    n_runs: int = 5,
) -> dict:
    """Run VAD ``n_runs`` times and return timing stats."""
    # Warm-up
    _ = vad.detect(waveform, SAMPLE_RATE)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        segments = vad.detect(waveform, SAMPLE_RATE)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    lat_ms = [l * 1000 for l in latencies]
    avg_lat = sum(lat_ms) / len(lat_ms)
    min_lat = min(lat_ms)
    max_lat = max(lat_ms)
    rtf = (sum(latencies) / len(latencies)) / audio_duration_sec

    return {
        "name": name,
        "audio_duration_sec": audio_duration_sec,
        "segments": segments,
        "n_segments": len(segments),
        "avg_latency_ms": avg_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "rtf": rtf,
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("EchoSelf VAD Benchmark  (Week 2 — Voice Activity Detection)")
    print("=" * 70)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Sample rate     : {SAMPLE_RATE} Hz")
    print()

    # Test scenarios
    scenarios = [
        ("Pure tone (1 s)",   _make_tone(1.0),              1.0),
        ("Silence (1 s)",     _make_silence(1.0),            1.0),
        ("Mixed (3 s)",       _make_mixed(1.0, 1.0),         3.0),
        ("Long tone (10 s)",  _make_tone(10.0),             10.0),
        ("Long mixed (30 s)", _make_mixed(10.0, 10.0),      30.0),
    ]

    methods = [
        ("energy", create_vad(method="energy", hangover_ms=200, min_speech_duration_ms=100)),
        ("neural", create_vad(method="neural", hangover_ms=200, min_speech_duration_ms=100)),
    ]

    for method_name, vad in methods:
        print(f"{'─' * 70}")
        print(f"Backend: {method_name.upper()}")
        print(f"{'─' * 70}")
        header = f"{'Scenario':<26}  {'Dur(s)':>6}  {'Segs':>5}  {'Avg(ms)':>9}  {'Min(ms)':>9}  {'RTF':>8}"
        print(header)
        print("─" * len(header))

        for scenario_name, waveform, duration in scenarios:
            result = _benchmark(method_name, vad, waveform, duration, n_runs=5)
            rtf_str = f"{result['rtf']:.4f}"
            rtf_note = " ✓" if result["rtf"] < 1.0 else " ✗"
            print(
                f"{scenario_name:<26}  "
                f"{result['audio_duration_sec']:>6.1f}  "
                f"{result['n_segments']:>5}  "
                f"{result['avg_latency_ms']:>9.2f}  "
                f"{result['min_latency_ms']:>9.2f}  "
                f"{rtf_str:>8}{rtf_note}"
            )
        print()

    print("=" * 70)
    print("Benchmark complete. RTF < 1.0 = faster than real-time (✓).")
    print()

    # Frame size comparison
    print("Frame size comparison (energy VAD, 1-s tone, 5 runs):")
    tone = _make_tone(1.0)
    for ms in (10, 20, 30):
        vad = create_vad(method="energy", frame_duration_ms=ms, hangover_ms=0, min_speech_duration_ms=0)
        result = _benchmark(f"{ms}ms", vad, tone, 1.0, n_runs=5)
        print(
            f"  frame={ms:2d}ms  avg={result['avg_latency_ms']:.2f}ms  "
            f"RTF={result['rtf']:.4f}  segs={result['n_segments']}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
