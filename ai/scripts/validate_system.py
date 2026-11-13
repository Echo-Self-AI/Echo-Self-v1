"""
EchoSelf Week 10 — System Validation & Performance Measurement
==============================================================
Runs comprehensive validation across all subsystems:
  1. VAD (EnergyVAD) performance
  2. NVS (Neural Voice Synthesis) latency + quality
  3. Speaker Verification accuracy
  4. Classical Watermarking SNR + robustness
  5. Learned Watermarking bit accuracy
  6. Full Pipeline end-to-end
  7. Resource measurement (memory, GPU availability)

Outputs:
  ai/experiments/validation_report.json
  ai/experiments/validation_watermark_snr.png
  ai/experiments/validation_latency.png
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import time
import tracemalloc
import wave
from pathlib import Path
from typing import Dict, Any, List

import torch

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

# Repo root is 3 levels up from scripts/
_HERE = Path(__file__).resolve().parent
_AI_DIR = _HERE.parent
_EXPERIMENTS_DIR = _AI_DIR / "experiments"
_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_system")

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen_silence(duration_s: float) -> torch.Tensor:
    n = int(duration_s * SAMPLE_RATE)
    return torch.zeros(n)


def _gen_tone(freq_hz: float, duration_s: float, amp: float = 0.8) -> torch.Tensor:
    n = int(duration_s * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    return amp * torch.sin(2 * math.pi * freq_hz * t)


def _header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _row(label: str, value: Any) -> None:
    print(f"  {label:<40} {value}")


# ---------------------------------------------------------------------------
# 1. VAD validation
# ---------------------------------------------------------------------------

def validate_vad() -> Dict:
    _header("1. VAD Validation")

    from echoself_ai.voice.vad import EnergyVAD
    from echoself_ai.voice.vad_config import VADConfig

    cfg = VADConfig(
        method="energy",
        energy_threshold=-4.0,
        min_speech_duration_ms=100,
        hangover_ms=200,
        sample_rate=SAMPLE_RATE,
    )
    vad = EnergyVAD(cfg)

    # --- Test signals ---
    silence_2s     = _gen_silence(2.0)
    tone_2s        = _gen_tone(440.0, 2.0)
    gap            = _gen_silence(1.0)
    tone_plus_gap  = torch.cat([_gen_tone(440.0, 2.0), gap, _gen_tone(440.0, 2.0)])

    results = {}

    # --- Pure silence ---
    t0 = time.perf_counter()
    segs_silence = vad.detect(silence_2s, SAMPLE_RATE)
    lat_silence = (time.perf_counter() - t0) * 1000
    fp_on_silence = len(segs_silence)
    _row("Silence 2s — segments found", len(segs_silence))
    _row("Silence 2s — false positives", fp_on_silence)
    _row("Silence 2s — latency (ms)", f"{lat_silence:.2f}")

    # --- Pure tone ---
    t0 = time.perf_counter()
    segs_tone = vad.detect(tone_2s, SAMPLE_RATE)
    lat_tone = (time.perf_counter() - t0) * 1000
    detection_rate = 1.0 if len(segs_tone) > 0 else 0.0
    _row("Tone 2s — segments found", len(segs_tone))
    _row("Tone 2s — detection rate", f"{detection_rate:.1%}")
    _row("Tone 2s — latency (ms)", f"{lat_tone:.2f}")

    # --- Tone + 1s silence + tone ---
    t0 = time.perf_counter()
    segs_gap = vad.detect(tone_plus_gap, SAMPLE_RATE)
    lat_gap = (time.perf_counter() - t0) * 1000
    _row("Tone+Gap+Tone — segments found", len(segs_gap))
    _row("Tone+Gap+Tone — latency (ms)", f"{lat_gap:.2f}")

    results = {
        "silence_segments": len(segs_silence),
        "silence_false_positives": fp_on_silence,
        "silence_latency_ms": round(lat_silence, 3),
        "tone_segments": len(segs_tone),
        "tone_detection_rate": detection_rate,
        "tone_latency_ms": round(lat_tone, 3),
        "tone_gap_tone_segments": len(segs_gap),
        "tone_gap_tone_latency_ms": round(lat_gap, 3),
    }
    return results


# ---------------------------------------------------------------------------
# 2. NVS validation
# ---------------------------------------------------------------------------

def validate_nvs() -> Dict:
    _header("2. NVS Validation")

    from echoself_ai.voice.nvs import TextToSpeech, NVSConfig

    tts = TextToSpeech(NVSConfig(sample_rate=SAMPLE_RATE))

    texts = ["hello", "echo self", "testing one two three", "voice synthesis", "neural audio"]
    latencies: List[float] = []
    per_utterance = []

    for text in texts:
        t0 = time.perf_counter()
        waveform, sr = tts.synthesize(text)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        duration_s = waveform.numel() / sr if waveform.numel() > 0 else 0.0
        peak_amp = float(waveform.abs().max()) if waveform.numel() > 0 else 0.0
        has_nan = bool(torch.isnan(waveform).any())
        has_inf = bool(torch.isinf(waveform).any())

        _row(f"  '{text}' — duration (s)", f"{duration_s:.3f}")
        _row(f"  '{text}' — peak amp", f"{peak_amp:.4f}")
        _row(f"  '{text}' — has NaN/Inf", f"{has_nan or has_inf}")
        _row(f"  '{text}' — latency (ms)", f"{lat:.1f}")

        per_utterance.append({
            "text": text,
            "duration_s": round(duration_s, 4),
            "peak_amplitude": round(peak_amp, 4),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "latency_ms": round(lat, 2),
        })

    mean_lat = sum(latencies) / len(latencies)
    std_lat  = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5

    _row("Mean latency (ms)", f"{mean_lat:.1f}")
    _row("Std  latency (ms)", f"{std_lat:.1f}")

    return {
        "per_utterance": per_utterance,
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
    }


# ---------------------------------------------------------------------------
# 3. Speaker Verification validation
# ---------------------------------------------------------------------------

def validate_speaker_verification() -> Dict:
    _header("3. Speaker Verification Validation")

    from echoself_ai.voice.speaker_verification import (
        SpeakerEncoder,
        SpeakerEnrollment,
        SpeakerVerifier,
    )

    # Use a fresh temp dir for this validation run
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())

    encoder = SpeakerEncoder()
    enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_dir)
    verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_dir)

    # Enroll with 3 waveforms of same frequency (same 'speaker')
    enroll_waveforms = [_gen_tone(300.0, 2.0) for _ in range(3)]
    profile = enrollment.enroll(
        user_id="validation_speaker",
        waveform_list=enroll_waveforms,
        sample_rate=SAMPLE_RATE,
    )
    _row("Enrolled speaker ID", profile.user_id)
    _row("Enrollment ID", profile.enrollment_id)

    # --- Matched: same frequency ---
    matched_waveform = _gen_tone(300.0, 2.0)
    accepted_match, sim_match = verifier.verify(
        "validation_speaker", matched_waveform, SAMPLE_RATE, threshold=0.0
    )
    _row("Matched freq  — similarity", f"{sim_match:.4f}")
    _row("Matched freq  — accepted (th=0.0)", accepted_match)

    # --- Unmatched: very different frequency ---
    unmatched_waveform = _gen_tone(4000.0, 2.0)
    accepted_unmatch, sim_unmatch = verifier.verify(
        "validation_speaker", unmatched_waveform, SAMPLE_RATE, threshold=0.0
    )
    _row("Unmatched freq — similarity", f"{sim_unmatch:.4f}")
    _row("Unmatched freq — accepted (th=0.0)", accepted_unmatch)

    _row("Matched sim vs Unmatched sim", f"{sim_match:.4f} > {sim_unmatch:.4f} → {sim_match > sim_unmatch}")

    # FAR proxy: fraction of unmatched that are accepted at default threshold (0.75)
    _, sim_m = verifier.verify("validation_speaker", matched_waveform, SAMPLE_RATE, threshold=0.75)
    _, sim_u = verifier.verify("validation_speaker", unmatched_waveform, SAMPLE_RATE, threshold=0.75)
    # single-sample proxies
    far_proxy = 1.0 if sim_u >= 0.75 else 0.0
    frr_proxy = 1.0 if sim_m < 0.75 else 0.0
    _row("FAR proxy (single unmatched sample)", far_proxy)
    _row("FRR proxy (single matched sample)", frr_proxy)

    return {
        "enrolled_samples": 3,
        "matched_similarity": round(float(sim_match), 4),
        "unmatched_similarity": round(float(sim_unmatch), 4),
        "matched_gt_unmatched": bool(sim_match > sim_unmatch),
        "far_proxy": far_proxy,
        "frr_proxy": frr_proxy,
        "mean_similarity_matched": round(float(sim_match), 4),
        "mean_similarity_unmatched": round(float(sim_unmatch), 4),
    }


# ---------------------------------------------------------------------------
# 4. Classical Watermarking validation
# ---------------------------------------------------------------------------

def validate_classical_watermark() -> Dict:
    _header("4. Classical Watermarking Validation")

    from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
    from echoself_ai.watermark.config import ClassicalWatermarkConfig

    audio = _gen_tone(440.0, 2.0)
    watermark_id = "validation_wm_id"

    strengths = [0.001, 0.01, 0.05]
    rows: List[Dict] = []

    _row("Strength", f"{'SNR(dB)':<12} {'Det.Rate':<12} {'Latency(ms)'}")
    print(f"  {'-'*50}")

    for strength in strengths:
        cfg = ClassicalWatermarkConfig(
            watermark_id=watermark_id,
            strength=strength,
            sample_rate=SAMPLE_RATE,
        )
        wm = SpreadSpectrumWatermarker(cfg)

        # Embed
        t0 = time.perf_counter()
        watermarked = wm.embed(audio, watermark_id)
        embed_lat = (time.perf_counter() - t0) * 1000

        # SNR
        snr = wm.compute_snr(audio, watermarked)

        # Detection accuracy over 10 trials
        n_trials = 10
        detections = 0
        t0 = time.perf_counter()
        for _ in range(n_trials):
            detected, corr, ber = wm.extract(watermarked, watermark_id)
            if detected:
                detections += 1
        detect_lat = (time.perf_counter() - t0) * 1000 / n_trials
        det_rate = detections / n_trials

        # Robustness: add Gaussian noise then re-detect
        noisy = watermarked + torch.randn_like(watermarked) * 0.01
        noisy = torch.clamp(noisy, -1.0, 1.0)
        noise_detections = 0
        for _ in range(n_trials):
            det_n, _, _ = wm.extract(noisy, watermark_id)
            if det_n:
                noise_detections += 1
        noise_success_rate = noise_detections / n_trials

        _row(f"  {strength:.3f}", f"{snr:>8.1f} dB  {det_rate:>8.1%}  {detect_lat:>8.2f} ms")
        _row(f"    → Gaussian noise success", f"{noise_success_rate:.1%}")

        rows.append({
            "strength": strength,
            "snr_db": round(float(snr), 2),
            "detection_rate": det_rate,
            "embed_latency_ms": round(embed_lat, 2),
            "detect_latency_ms": round(detect_lat, 2),
            "noise_success_rate": noise_success_rate,
        })

    return {"classical_watermark": rows}


# ---------------------------------------------------------------------------
# 5. Learned Watermarking validation
# ---------------------------------------------------------------------------

def validate_learned_watermark() -> Dict:
    _header("5. Learned Watermarking Validation")

    from echoself_ai.watermark.learned import LearnedWatermarker, LearnedWatermarkConfig

    config = LearnedWatermarkConfig(
        n_bits=16,
        hidden_dim=64,
        segment_length=SAMPLE_RATE,  # 1s
    )

    # Use a temp checkpoint dir
    import tempfile
    tmp_dir = tempfile.mkdtemp()

    wm = LearnedWatermarker(config=config, checkpoint_dir=tmp_dir)

    _row("Training for 100 steps …", "")
    t0 = time.perf_counter()
    train_result = wm.train_model(n_steps=100, batch_size=4, learning_rate=1e-3)
    train_lat = (time.perf_counter() - t0) * 1000
    _row("Training latency (ms)", f"{train_lat:.0f}")
    _row("Final training loss", f"{train_result['final_loss']:.4f}")

    # Embed and extract on clean 1s audio
    audio_1s = _gen_tone(440.0, 1.0)
    bits_in = torch.randint(0, 2, (config.n_bits,)).float()

    watermarked = wm.embed(audio_1s, bits_in)
    bit_probs = wm.extract(watermarked)

    # Bit accuracy: compare hard decisions
    hard_out = (bit_probs >= 0.5).float()
    bit_acc = (hard_out == bits_in).float().mean().item()
    _row("Bit accuracy (clean)", f"{bit_acc:.3f}")
    _row("Above 50% baseline", bit_acc > 0.5)

    # Compare classical vs learned
    watermark_id = "validation_learned_cmp"
    comparison = wm.compare_with_classical(audio_1s, watermark_id)
    _row("Classical detected", comparison["classical"]["detected"])
    _row("Classical correlation", f"{comparison['classical']['correlation']:.4f}")
    _row("Learned bit accuracy", f"{comparison['learned']['bit_accuracy']:.3f}")

    # Tabulate comparison
    print(f"\n  {'Method':<20} {'Metric':<30} {'Value'}")
    print(f"  {'-'*60}")
    print(f"  {'Classical':<20} {'detected':<30} {comparison['classical']['detected']}")
    print(f"  {'Classical':<20} {'correlation':<30} {comparison['classical']['correlation']:.4f}")
    print(f"  {'Classical':<20} {'bit_error_rate':<30} {comparison['classical']['ber']:.4f}")
    print(f"  {'Learned':<20} {'bit_accuracy':<30} {comparison['learned']['bit_accuracy']:.4f}")
    print(f"  {'Learned':<20} {'mean_confidence':<30} {comparison['learned']['mean_confidence']:.4f}")

    return {
        "train_steps": 100,
        "final_loss": round(float(train_result["final_loss"]), 4),
        "bit_accuracy_clean": round(float(bit_acc), 4),
        "above_50pct_baseline": bit_acc > 0.5,
        "comparison": {
            "classical_detected": comparison["classical"]["detected"],
            "classical_correlation": round(comparison["classical"]["correlation"], 4),
            "classical_ber": round(comparison["classical"]["ber"], 4),
            "learned_bit_accuracy": round(comparison["learned"]["bit_accuracy"], 4),
            "learned_mean_confidence": round(comparison["learned"]["mean_confidence"], 4),
        },
    }


# ---------------------------------------------------------------------------
# 6. Full Pipeline validation
# ---------------------------------------------------------------------------

def validate_pipeline() -> Dict:
    _header("6. Full Pipeline Validation")

    from echoself_ai.pipeline import SpeechPipeline, PipelineConfig
    from echoself_ai.voice.audio import generate_synthetic_tone

    cfg = PipelineConfig(
        sample_rate=SAMPLE_RATE,
        vad_enabled=True,
        asr_enabled=True,
        nvs_enabled=True,
        watermark_enabled=True,
        audit_enabled=True,
    )

    pipeline = SpeechPipeline(cfg)

    # Synthetic audio input
    audio_input = generate_synthetic_tone(freq_hz=440.0, duration_seconds=2.0,
                                          sample_rate=SAMPLE_RATE)

    t0 = time.perf_counter()
    result = pipeline.process(
        input_waveform=audio_input,
        sample_rate=SAMPLE_RATE,
        text_input="hello world",
    )
    total_wall = (time.perf_counter() - t0) * 1000

    _row("Pipeline success", result.success)
    _row("Job ID", result.job_id)
    _row("Input duration (s)", f"{result.input_duration_seconds:.3f}")
    _row("Total latency (ms)", f"{result.total_latency_ms:.1f}")
    _row("Real-time factor", f"{result.real_time_factor:.4f}")
    _row("Watermark detected", result.detection_result)

    if result.error:
        _row("Error", result.error)

    # Stage latencies
    print(f"\n  Stage Latencies:")
    for stage, lat in result.stage_latencies.items():
        _row(f"    {stage}", f"{lat:.2f} ms")

    # Validate output WAV
    wav_valid = False
    if result.output_waveform is not None and result.output_waveform.numel() > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name

        from echoself_ai.voice.audio import write_wav
        write_wav(tmp_wav, result.output_waveform, result.output_sample_rate or SAMPLE_RATE)
        try:
            with wave.open(tmp_wav, "rb") as wf:
                nframes = wf.getnframes()
            wav_valid = nframes > 0
        finally:
            os.unlink(tmp_wav)

    _row("Output WAV valid", wav_valid)

    return {
        "success": result.success,
        "error": result.error,
        "input_duration_s": result.input_duration_seconds,
        "total_latency_ms": round(result.total_latency_ms, 2),
        "real_time_factor": round(result.real_time_factor, 4),
        "watermark_detected": result.detection_result,
        "output_wav_valid": wav_valid,
        "stage_latencies": {k: round(v, 2) for k, v in result.stage_latencies.items()},
    }


# ---------------------------------------------------------------------------
# 7. Resource measurement
# ---------------------------------------------------------------------------

def measure_resources() -> Dict:
    _header("7. Resource Measurement")

    # Peak memory via tracemalloc
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Trigger a small workload
    from echoself_ai.voice.nvs import TextToSpeech, NVSConfig
    tts = TextToSpeech(NVSConfig(sample_rate=SAMPLE_RATE))
    waveform, _ = tts.synthesize("memory test")
    del waveform

    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    peak_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_traced_mb = peak / (1024 * 1024)

    # GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    mode = "GPU" if gpu_available else "CPU"

    _row("Peak traced memory (MB)", f"{peak_traced_mb:.2f}")
    _row("GPU available", gpu_available)
    _row("GPU count", gpu_count)
    _row("GPU name", gpu_name)
    _row("Running mode", mode)

    return {
        "peak_traced_memory_mb": round(peak_traced_mb, 2),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# 8. Generate charts
# ---------------------------------------------------------------------------

def generate_charts(classical_data: List[Dict], pipeline_latencies: Dict) -> None:
    _header("8. Generating Charts")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping chart generation.")
        return

    # --- Watermark SNR chart ---
    strengths = [str(r["strength"]) for r in classical_data]
    snrs = [r["snr_db"] for r in classical_data]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(strengths, snrs, color=["#4c8cbf", "#2d6a9f", "#1a4b7a"], edgecolor="black")
    ax.set_xlabel("Watermark Strength (alpha)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Classical Watermark: SNR vs Strength")
    for bar, val in zip(bars, snrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out_snr = _EXPERIMENTS_DIR / "validation_watermark_snr.png"
    plt.savefig(str(out_snr), dpi=120)
    plt.close()
    _row("Saved SNR chart", str(out_snr))

    # --- Pipeline latency chart ---
    if pipeline_latencies:
        stages = list(pipeline_latencies.keys())
        lats = list(pipeline_latencies.values())

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = plt.cm.Blues([0.5 + 0.4 * i / max(len(stages) - 1, 1) for i in range(len(stages))])
        bars = ax.bar(stages, lats, color=colors, edgecolor="black")
        ax.set_xlabel("Pipeline Stage")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Full Pipeline Stage Latencies")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, lats):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        out_lat = _EXPERIMENTS_DIR / "validation_latency.png"
        plt.savefig(str(out_lat), dpi=120)
        plt.close()
        _row("Saved latency chart", str(out_lat))


# ---------------------------------------------------------------------------
# 9. Write JSON report
# ---------------------------------------------------------------------------

def write_report(report: Dict) -> None:
    out = _EXPERIMENTS_DIR / "validation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    _row("Report written to", str(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  EchoSelf Week 10 — System Validation")
    print("=" * 60)

    report: Dict[str, Any] = {}

    # 1. VAD
    report["vad"] = validate_vad()

    # 2. NVS
    report["nvs"] = validate_nvs()

    # 3. Speaker verification
    report["speaker_verification"] = validate_speaker_verification()

    # 4. Classical watermark
    cw_result = validate_classical_watermark()
    report["classical_watermark"] = cw_result["classical_watermark"]

    # 5. Learned watermark
    report["learned_watermark"] = validate_learned_watermark()

    # 6. Full pipeline
    report["pipeline"] = validate_pipeline()

    # 7. Resources
    report["resources"] = measure_resources()

    # 8. Charts
    generate_charts(
        classical_data=report["classical_watermark"],
        pipeline_latencies=report["pipeline"].get("stage_latencies", {}),
    )

    # 9. JSON report
    _header("9. Summary Report")
    write_report(report)

    # Final summary
    _header("VALIDATION COMPLETE")
    _row("VAD false positives on silence", report["vad"]["silence_false_positives"])
    _row("VAD tone detection rate", f"{report['vad']['tone_detection_rate']:.1%}")
    _row("NVS mean latency (ms)", report["nvs"]["mean_latency_ms"])
    _row("Speaker match>unmatch", report["speaker_verification"]["matched_gt_unmatched"])
    best_snr = max(r["snr_db"] for r in report["classical_watermark"])
    _row("Best classical SNR (dB)", f"{best_snr:.1f}")
    _row("Learned watermark bit acc", report["learned_watermark"]["bit_accuracy_clean"])
    _row("Pipeline success", report["pipeline"]["success"])
    _row("Pipeline RTF", report["pipeline"]["real_time_factor"])
    _row("Mode", report["resources"]["mode"])


if __name__ == "__main__":
    main()
