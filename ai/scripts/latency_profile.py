#!/usr/bin/env python3
"""EchoSelf Week 9 — End-to-End Speech Pipeline Latency Profiler.

Instantiates a SpeechPipeline, processes synthetic audio of three durations
(1 s, 2 s, 5 s), runs run_latency_profile(n_trials=3), and prints a full
latency table (stage, mean ms, real-time factor).

Usage::

    cd /home/raghu/Documents/projects/echoself/ai
    .venv/bin/python scripts/latency_profile.py
"""
from __future__ import annotations

import sys
import textwrap

import torch

# Ensure the package is importable when run from ai/ directory
sys.path.insert(0, "src")

from echoself_ai.pipeline import SpeechPipeline, PipelineConfig
from echoself_ai.voice.audio import generate_synthetic_tone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLUMN_WIDTH = 20

def _header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_latency_row(stage: str, mean_ms: float, rtf: float) -> None:
    stage_col = stage.ljust(COLUMN_WIDTH)
    mean_col = f"{mean_ms:>10.2f} ms"
    rtf_col = f"{rtf:>12.4f}"
    print(f"  {stage_col}  {mean_col}  {rtf_col}")


def _print_latency_table_header() -> None:
    stage_col = "Stage".ljust(COLUMN_WIDTH)
    mean_col = "Mean latency".rjust(13)
    rtf_col = "Stage RTF".rjust(13)
    print(f"  {stage_col}  {mean_col}  {rtf_col}")
    print("  " + "-" * (COLUMN_WIDTH + 30))


def _run_single(
    pipeline: SpeechPipeline,
    duration_seconds: float,
    sample_rate: int = 16000,
    label: str = "",
) -> None:
    """Run the pipeline on a synthetic tone of the given duration and print results."""
    tone = generate_synthetic_tone(
        freq_hz=440.0,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
    )

    _header(f"Single-shot: {label or f'{duration_seconds}s synthetic tone'}")

    result = pipeline.process(tone, sample_rate=sample_rate, text_input="hello echoself")

    print(f"  Job ID            : {result.job_id}")
    print(f"  Input duration    : {result.input_duration_seconds:.3f} s")
    print(f"  Total latency     : {result.total_latency_ms:.2f} ms")
    print(f"  Real-time factor  : {result.real_time_factor:.4f}")
    print(f"  Success           : {result.success}")
    print(f"  Transcript        : {result.transcript!r}")
    print(f"  Watermark ID      : {result.watermark_id}")
    print(f"  Detection result  : {result.detection_result}")
    print(f"  Audit event ID    : {result.audit_event_id}")
    if result.error:
        print(f"  Error             : {result.error}")

    print()
    print("  Per-stage latencies:")
    _print_latency_table_header()

    input_ms = result.input_duration_seconds * 1000.0
    for stage, lat_ms in sorted(result.stage_latencies.items()):
        stage_rtf = lat_ms / input_ms if input_ms > 0 else 0.0
        _print_latency_row(stage, lat_ms, stage_rtf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("EchoSelf Week 9 — Speech Pipeline Latency Profiler")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)

    # -------------------------------------------------------------------
    # 1. Instantiate pipeline
    # -------------------------------------------------------------------
    _header("Instantiating SpeechPipeline …")
    cfg = PipelineConfig(sample_rate=16000)
    pipeline = SpeechPipeline(cfg)
    print("  SpeechPipeline ready.")

    # -------------------------------------------------------------------
    # 2. Single-shot runs at 1 s, 2 s, 5 s
    # -------------------------------------------------------------------
    for dur in [1.0, 2.0, 5.0]:
        _run_single(pipeline, duration_seconds=dur, sample_rate=16000)

    # -------------------------------------------------------------------
    # 3. run_latency_profile(n_trials=3)
    # -------------------------------------------------------------------
    _header("run_latency_profile(n_trials=3) …")
    print("  Running 3 trials — this may take a moment …")
    profile = pipeline.run_latency_profile(n_trials=3)

    print()
    print(f"  Trials completed  : {profile['n_trials']}")
    print(f"  Mean total latency: {profile['total_mean_ms']:.2f} ms")
    print(f"  Mean RTF          : {profile['rtf_mean']:.4f}")
    print(f"  Std  RTF          : {profile['rtf_std']:.4f}")

    print()
    print("  Full per-stage latency table (over 3 trials, 1-second probe):")
    print()

    # Header row
    stage_col  = "Stage".ljust(COLUMN_WIDTH)
    mean_col   = "Mean ms".rjust(10)
    std_col    = "Std ms".rjust(10)
    min_col    = "Min ms".rjust(10)
    max_col    = "Max ms".rjust(10)
    rtf_col    = "Stage RTF".rjust(12)
    print(f"  {stage_col}  {mean_col}  {std_col}  {min_col}  {max_col}  {rtf_col}")
    print("  " + "-" * (COLUMN_WIDTH + 58))

    stages = sorted(profile["mean"].keys())
    # Use a 1-second probe for RTF normalisation
    input_ms_1s = 1000.0

    for stage in stages:
        mean_ms = profile["mean"].get(stage, 0.0)
        std_ms  = profile["std"].get(stage, 0.0)
        min_ms  = profile["min"].get(stage, 0.0)
        max_ms  = profile["max"].get(stage, 0.0)
        s_rtf   = mean_ms / input_ms_1s

        stage_c = stage.ljust(COLUMN_WIDTH)
        print(
            f"  {stage_c}  {mean_ms:>10.2f}  {std_ms:>10.2f}"
            f"  {min_ms:>10.2f}  {max_ms:>10.2f}  {s_rtf:>12.4f}"
        )

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
