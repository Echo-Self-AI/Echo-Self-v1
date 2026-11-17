#!/usr/bin/env python3
"""
EchoSelf — Interactive Demo
============================
Demonstrates the full pipeline live:
  1. Speaker enrollment
  2. Speaker verification
  3. Voice synthesis (text → WAV)
  4. Watermark embed + detect
  5. Audit log inspection
  6. End-to-end pipeline run

Usage:
    python demo.py              # interactive menu
    python demo.py --all        # run all demos automatically
    python demo.py --step 3     # run a specific step (1-6)

Output WAV files are written to:  demo_output/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Make sure the package is importable when running from ai/ ────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

# ── Colour helpers ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"


def title(text: str) -> None:
    width = 60
    print()
    print(BOLD + CYAN + "┌" + "─" * (width - 2) + "┐" + RESET)
    print(BOLD + CYAN + "│" + RESET + BOLD + f"  {text}".ljust(width - 2) + CYAN + "│" + RESET)
    print(BOLD + CYAN + "└" + "─" * (width - 2) + "┘" + RESET)


def ok(text: str) -> None:
    print(GREEN + "  ✓ " + RESET + text)


def info(text: str) -> None:
    print(CYAN + "  › " + RESET + text)


def warn(text: str) -> None:
    print(YELLOW + "  ⚠ " + RESET + text)


def err(text: str) -> None:
    print(RED + "  ✗ " + RESET + text)


def separator() -> None:
    print(DIM + "  " + "─" * 56 + RESET)


# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "demo_output"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Step 1 — Speaker Enrollment
# ============================================================================

def demo_enrollment() -> None:
    title("Step 1 · Speaker Enrollment")

    from echoself_ai.voice.speaker_verification import (
        SpeakerEnrollment,
        SpeakerVerifier,
        generate_challenge_phrase,
    )
    from echoself_ai.voice.audio import generate_synthetic_tone

    phrase = generate_challenge_phrase()
    info(f"Challenge phrase: \"{phrase}\"")
    info("Generating synthetic enrollment audio (3 samples at 440 Hz)…")

    samples = [generate_synthetic_tone(440.0, duration_seconds=2.0 + i * 0.5) for i in range(3)]

    enrollment = SpeakerEnrollment()
    t0 = time.perf_counter()
    profile = enrollment.enroll("demo_user", samples, sample_rate=16000)
    elapsed = (time.perf_counter() - t0) * 1000

    ok(f"User enrolled  — ID: {profile.enrollment_id[:16]}…")
    ok(f"Embedding dim  — {len(profile.mean_embedding)}")
    ok(f"Enroll latency — {elapsed:.1f} ms")
    ok(f"Profile saved  — data/speakers/demo_user.json")


# ============================================================================
# Step 2 — Speaker Verification
# ============================================================================

def demo_verification() -> None:
    title("Step 2 · Speaker Verification")

    from echoself_ai.voice.speaker_verification import SpeakerVerifier, SpeakerEnrollment
    from echoself_ai.voice.audio import generate_synthetic_tone

    # Ensure a profile exists
    enrollment = SpeakerEnrollment()
    if enrollment.load_profile("demo_user") is None:
        warn("No profile found — running enrollment first.")
        demo_enrollment()

    verifier = SpeakerVerifier()

    # Matched speaker (same 440 Hz)
    t0 = time.perf_counter()
    accepted, similarity = verifier.verify(
        "demo_user",
        generate_synthetic_tone(440.0, duration_seconds=1.5),
        sample_rate=16000,
        threshold=0.0,          # threshold=0 so result shows real similarity value
    )
    ms = (time.perf_counter() - t0) * 1000

    info("Test A — same speaker (440 Hz, threshold disabled to show raw similarity):")
    ok(f"  similarity = {similarity:.4f}   latency = {ms:.1f} ms")

    # Different speaker (880 Hz)
    _, sim_diff = verifier.verify(
        "demo_user",
        generate_synthetic_tone(880.0, duration_seconds=1.5),
        sample_rate=16000,
        threshold=0.0,
    )
    info("Test B — different speaker (880 Hz):")
    ok(f"  similarity = {sim_diff:.4f}")

    separator()
    info("Note: speaker encoder uses random (untrained) weights in this demo.")
    info("With trained weights, same-speaker similarity would be >0.90.")


# ============================================================================
# Step 3 — Voice Synthesis
# ============================================================================

def demo_synthesis() -> None:
    title("Step 3 · Neural Voice Synthesis")

    from echoself_ai.voice.nvs import TextToSpeech, NVSConfig
    from echoself_ai.voice.audio import write_wav

    config = NVSConfig()
    tts = TextToSpeech(config)

    phrases = [
        "Hello, I am Echo Self.",
        "Your voice and identity, echoed digitally.",
        "Speech synthesis complete.",
    ]

    for i, text in enumerate(phrases):
        t0 = time.perf_counter()
        waveform, sr = tts.synthesize(text)
        ms = (time.perf_counter() - t0) * 1000

        duration = len(waveform) / sr
        peak = waveform.abs().max().item()
        has_nan = not torch.isfinite(waveform).all()

        out_path = OUT_DIR / f"synthesis_{i+1}.wav"
        write_wav(out_path, waveform, sr)

        ok(f"\"{text}\"")
        info(f"  → {out_path.name}  |  {duration:.2f}s  |  peak={peak:.3f}  |  {ms:.0f} ms  |  NaN={has_nan}")

    separator()
    ok(f"All WAV files written to {OUT_DIR}/")


# ============================================================================
# Step 4 — Watermark Embed + Detect
# ============================================================================

def demo_watermark() -> None:
    title("Step 4 · Audio Watermarking")

    from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
    from echoself_ai.watermark.config import ClassicalWatermarkConfig
    from echoself_ai.watermark.detector import WatermarkDetector
    from echoself_ai.voice.audio import generate_synthetic_tone, write_wav
    import uuid

    watermark_id = str(uuid.uuid4())
    info(f"Watermark ID: {watermark_id}")

    config = ClassicalWatermarkConfig(watermark_id=watermark_id, strength=0.01)
    embedder = WatermarkDetector()
    wm = SpreadSpectrumWatermarker(config)

    # Generate source audio
    source = generate_synthetic_tone(440.0, duration_seconds=3.0)

    # Embed
    t0 = time.perf_counter()
    watermarked = wm.embed(source, watermark_id)
    embed_ms = (time.perf_counter() - t0) * 1000

    # SNR
    snr = wm.compute_snr(source, watermarked)
    ok(f"Embedded  — SNR = {snr:.2f} dB   latency = {embed_ms:.2f} ms")

    # Detect on watermarked audio
    t0 = time.perf_counter()
    detected, correlation, ber = wm.extract(watermarked, watermark_id)
    detect_ms = (time.perf_counter() - t0) * 1000
    ok(f"Detected  — {detected}   correlation = {correlation:.5f}   BER = {ber:.3f}   latency = {detect_ms:.2f} ms")

    # Detect on clean (should be not detected)
    detected_clean, corr_clean, _ = wm.extract(source, watermark_id)
    ok(f"Clean audio — detected = {detected_clean}   correlation = {corr_clean:.5f}")

    # Write comparison pair
    write_wav(OUT_DIR / "watermark_source.wav", source, 16000)
    write_wav(OUT_DIR / "watermark_embedded.wav", watermarked, 16000)

    separator()

    # Robustness test: add noise
    info("Robustness: Gaussian noise (σ=0.01) →")
    noisy = watermarked + torch.randn_like(watermarked) * 0.01
    noisy = torch.clamp(noisy, -1.0, 1.0)
    det_noisy, corr_noisy, _ = wm.extract(noisy, watermark_id)
    ok(f"  After noise — detected = {det_noisy}   correlation = {corr_noisy:.5f}")

    info("Robustness: amplitude scale ×0.5 →")
    scaled = watermarked * 0.5
    det_scaled, corr_scaled, _ = wm.extract(scaled, watermark_id)
    ok(f"  After scaling — detected = {det_scaled}   correlation = {corr_scaled:.5f}")


# ============================================================================
# Step 5 — Audit Log
# ============================================================================

def demo_audit() -> None:
    title("Step 5 · Audit Logging")

    from echoself_ai.audit.logger import AuditLogger, AuditRecord, compute_audio_hash
    from echoself_ai.voice.audio import generate_synthetic_tone
    import uuid, datetime

    logger = AuditLogger()

    # Log a few synthetic events
    waveform = generate_synthetic_tone(440.0, 1.0)
    audio_hash = compute_audio_hash(waveform, 16000)

    events = [
        ("synthesize", "demo_user", "nvs-acoustic-v1", "classical", str(uuid.uuid4())),
        ("detect",     "demo_user", "detector-v1",     "classical", str(uuid.uuid4())),
        ("verify",     "demo_user", "speaker-v1",      "none",      None),
    ]

    for action, user, model, wm_type, wm_id in events:
        record = AuditRecord(
            event_id=str(uuid.uuid4()),
            user_id=user,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_id=model,
            watermark_type=wm_type,
            watermark_id=wm_id,
            output_hash=audio_hash,
            detection_result=True if action == "detect" else None,
            verification_status=True if action == "verify" else None,
            action=action,
            metadata={"demo": True},
        )
        logger.log(record)
        ok(f"Logged [{action}]  event_id={record.event_id[:16]}…")

    separator()

    # Read back
    all_records = logger.read_all()
    info(f"Total records in log: {len(all_records)}")

    # Integrity check
    ok(f"Integrity check: {logger.verify_integrity()}")
    info(f"Log file: data/audit/audit_log.jsonl")

    # Show last record
    if all_records:
        last = all_records[-1]
        info(f"Last record — action={last.action}, user={last.user_id}, model={last.model_id}")


# ============================================================================
# Step 6 — End-to-End Pipeline
# ============================================================================

def demo_pipeline() -> None:
    title("Step 6 · End-to-End Pipeline")

    from echoself_ai.pipeline import SpeechPipeline, PipelineConfig
    from echoself_ai.voice.audio import generate_synthetic_tone, write_wav

    config = PipelineConfig(
        vad_enabled=True,
        asr_enabled=True,
        nvs_enabled=True,
        watermark_enabled=True,
        audit_enabled=True,
    )
    pipeline = SpeechPipeline(config)

    info("Input: 2s synthetic tone + text_input 'Echo self demo, one two three'")
    input_wave = generate_synthetic_tone(freq_hz=440.0, duration_seconds=2.0)

    t0 = time.perf_counter()
    result = pipeline.process(
        input_wave,
        sample_rate=16000,
        text_input="Echo self demo, one two three.",
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    separator()

    if result.success:
        ok(f"Pipeline SUCCESS")
    else:
        err(f"Pipeline error: {result.error}")

    info(f"Job ID           : {result.job_id}")
    info(f"Transcript       : {result.transcript!r}")
    info(f"Watermark ID     : {result.watermark_id}")
    info(f"Watermark detect : {result.detection_result}")
    info(f"Audit event ID   : {result.audit_event_id}")

    separator()
    info("Stage latencies:")
    for stage, ms in sorted(result.stage_latencies.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(ms / 2))
        print(f"    {stage:<12} {ms:>7.2f} ms  {bar}")

    separator()
    info(f"Total latency    : {result.total_latency_ms:.2f} ms  (wall: {wall_ms:.0f} ms)")
    info(f"Real-time factor : {result.real_time_factor:.4f}  ({1/max(result.real_time_factor, 1e-9):.0f}× faster than real-time)")

    # Save output audio
    if result.output_waveform is not None:
        out_path = OUT_DIR / "pipeline_output.wav"
        write_wav(out_path, result.output_waveform, result.output_sample_rate or 16000)
        ok(f"Output WAV       : {out_path}")


# ============================================================================
# Main menu
# ============================================================================

STEPS = {
    1: ("Speaker Enrollment",     demo_enrollment),
    2: ("Speaker Verification",   demo_verification),
    3: ("Voice Synthesis",        demo_synthesis),
    4: ("Audio Watermarking",     demo_watermark),
    5: ("Audit Logging",          demo_audit),
    6: ("End-to-End Pipeline",    demo_pipeline),
}


def run_step(n: int) -> None:
    if n not in STEPS:
        err(f"Unknown step {n}. Choose 1–6.")
        return
    name, fn = STEPS[n]
    try:
        fn()
    except Exception as exc:
        err(f"Step {n} ({name}) raised: {exc}")
        import traceback
        traceback.print_exc()


def print_menu() -> None:
    print()
    print(BOLD + "  EchoSelf Demo — Select a step:" + RESET)
    print()
    for n, (name, _) in STEPS.items():
        print(f"    [{n}] {name}")
    print(f"    [a] Run all steps")
    print(f"    [q] Quit")
    print()


def interactive_loop() -> None:
    print()
    print(BOLD + CYAN + "  ███████╗ ██████╗██╗  ██╗ ██████╗ ███████╗███████╗██╗     ███████╗" + RESET)
    print(BOLD + CYAN + "  ██╔════╝██╔════╝██║  ██║██╔═══██╗██╔════╝██╔════╝██║     ██╔════╝" + RESET)
    print(BOLD + CYAN + "  █████╗  ██║     ███████║██║   ██║███████╗█████╗  ██║     █████╗  " + RESET)
    print(BOLD + CYAN + "  ██╔══╝  ██║     ██╔══██║██║   ██║╚════██║██╔══╝  ██║     ██╔══╝  " + RESET)
    print(BOLD + CYAN + "  ███████╗╚██████╗██║  ██║╚██████╔╝███████║███████╗███████╗██║     " + RESET)
    print(BOLD + CYAN + "  ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝     " + RESET)
    print()
    print(DIM + "  Your Voice and Identity, Echoed Digitally." + RESET)
    print(DIM + "  Local-first · Privacy-preserving · Free tools only · ₹0 cost" + RESET)
    print()

    while True:
        print_menu()
        try:
            choice = input("  › ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            info("Exiting.")
            break

        if choice == "q":
            info("Exiting.")
            break
        elif choice == "a":
            for n in STEPS:
                run_step(n)
        elif choice.isdigit() and int(choice) in STEPS:
            run_step(int(choice))
        else:
            warn("Invalid choice. Enter 1–6, 'a', or 'q'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoSelf demo — showcase all pipeline components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all",  action="store_true", help="Run all 6 demo steps without prompting")
    parser.add_argument("--step", type=int, choices=range(1, 7), metavar="N", help="Run a single step (1-6)")
    args = parser.parse_args()

    if args.all:
        # Banner
        print(BOLD + CYAN + "\n  EchoSelf Demo — Running all steps\n" + RESET)
        for n in STEPS:
            run_step(n)
        print()
        ok("All demo steps complete.")
        print(f"\n  Output files in: {OUT_DIR}/\n")
    elif args.step:
        run_step(args.step)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
