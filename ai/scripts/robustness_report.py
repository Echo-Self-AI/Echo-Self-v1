#!/usr/bin/env python
"""
EchoSelf Week 11 — Robustness Report Generator
===============================================

Runs all robustness checks programmatically, collects results into a dict,
writes ai/experiments/robustness_report.json, and prints a summary table.

Usage:
    cd ai/
    .venv/bin/python scripts/robustness_report.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_AI_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_AI_DIR / "src"))

SAMPLE_RATE = 16000
EXPERIMENTS_DIR = _AI_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------

def _silence(n_samples: int) -> torch.Tensor:
    return torch.zeros(n_samples, dtype=torch.float32)


def _tone(freq_hz: float, n_samples: int, amp: float = 0.5) -> torch.Tensor:
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    return amp * torch.sin(2.0 * math.pi * freq_hz * t)


def _resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    new_len = int(len(waveform) * target_sr / orig_sr)
    wav_3d = waveform.unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(wav_3d, size=new_len, mode="linear", align_corners=True)
    return resampled.squeeze(0).squeeze(0)


def _make_watermarker(strength: float = 0.01, watermark_id: str = "report_wm"):
    from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
    from echoself_ai.watermark.config import ClassicalWatermarkConfig

    cfg = ClassicalWatermarkConfig(
        watermark_id=watermark_id,
        strength=strength,
        sample_rate=SAMPLE_RATE,
    )
    return SpreadSpectrumWatermarker(cfg)


def _safe_run(fn, *args, **kwargs) -> Tuple[Any, str | None]:
    """Run fn(*args, **kwargs), return (result, None) or (None, error_str)."""
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc(limit=3)


# ===========================================================================
# SECTION 1 — Watermark Robustness
# ===========================================================================

def run_watermark_robustness() -> List[Dict[str, Any]]:
    """Run all watermark-robustness transforms, return list of result dicts."""
    WATERMARK_ID = "report_wm_id"
    STRENGTH = 0.01
    results: List[Dict[str, Any]] = []

    def embed_transform_extract(transform_fn, label: str) -> Dict[str, Any]:
        wm = _make_watermarker(STRENGTH, WATERMARK_ID)
        original = _tone(440.0, SAMPLE_RATE * 2)
        watermarked = wm.embed(original, WATERMARK_ID)

        try:
            transformed = transform_fn(watermarked)
            detected, corr, ber = wm.extract(transformed, WATERMARK_ID)
            return {
                "transform": label,
                "correlation": round(corr, 6),
                "detected": detected,
                "bit_error_rate": round(ber, 4),
                "error": None,
            }
        except Exception:
            return {
                "transform": label,
                "correlation": None,
                "detected": False,
                "bit_error_rate": None,
                "error": traceback.format_exc(limit=2),
            }

    # Gaussian noise
    for sigma in [0.001, 0.01, 0.1]:
        def _gaussian(w, s=sigma):
            return w + s * torch.randn_like(w)
        results.append(embed_transform_extract(_gaussian, f"gaussian_noise_sigma={sigma}"))

    # Amplitude scaling
    for scale in [0.5, 2.0]:
        def _scale(w, s=scale):
            return torch.clamp(w * s, -1.0, 1.0)
        results.append(embed_transform_extract(_scale, f"amplitude_scale_{scale}x"))

    # Trimming — remove first 10 %
    def _trim(w):
        cut = max(1, int(0.10 * w.numel()))
        return w[cut:]
    results.append(embed_transform_extract(_trim, "trim_first_10pct"))

    # Padding — 500 silent samples each side
    def _pad(w):
        pad = torch.zeros(500, dtype=w.dtype)
        return torch.cat([pad, w, pad])
    results.append(embed_transform_extract(_pad, "pad_500_samples_each_side"))

    # Resampling 16k → 32k → 16k
    def _resample(w):
        up = _resample_linear(w, 16000, 32000)
        down = _resample_linear(up, 32000, 16000)
        min_len = min(w.numel(), down.numel())
        return down[:min_len]
    results.append(embed_transform_extract(_resample, "resample_16k_32k_16k"))

    # DC offset +0.01
    def _dc(w):
        return torch.clamp(w + 0.01, -1.0, 1.0)
    results.append(embed_transform_extract(_dc, "dc_offset_+0.01"))

    return results


# ===========================================================================
# SECTION 2 — Speaker Verification Robustness
# ===========================================================================

def run_speaker_verification_robustness() -> List[Dict[str, Any]]:
    from echoself_ai.voice.speaker_verification import (
        SpeakerEncoder,
        SpeakerEnrollment,
        SpeakerVerifier,
    )

    results: List[Dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp())

    encoder = SpeakerEncoder()
    enrollment = SpeakerEnrollment(encoder=encoder, storage_dir=tmp_dir)
    verifier = SpeakerVerifier(encoder=encoder, storage_dir=tmp_dir)

    waveforms = [_tone(300.0, SAMPLE_RATE * 2) for _ in range(3)]
    enrollment.enroll("sv_report_user", waveforms, SAMPLE_RATE)

    def _verify(label: str, wav: torch.Tensor, user_id: str = "sv_report_user") -> Dict:
        try:
            accepted, sim = verifier.verify(user_id, wav, SAMPLE_RATE, threshold=0.0)
            return {"scenario": label, "accepted": accepted, "similarity": round(sim, 4), "error": None}
        except Exception:
            return {"scenario": label, "accepted": None, "similarity": None, "error": traceback.format_exc(limit=2)}

    results.append(_verify("same_waveform", _tone(300.0, SAMPLE_RATE * 2)))
    results.append(_verify("same_freq_half_amplitude", _tone(300.0, SAMPLE_RATE * 2, amp=0.25)))
    results.append(_verify("different_frequency_4000Hz", _tone(4000.0, SAMPLE_RATE * 2)))

    noisy = _tone(300.0, SAMPLE_RATE * 2) + 0.1 * torch.randn(SAMPLE_RATE * 2)
    results.append(_verify("added_noise_sigma=0.1", torch.clamp(noisy, -1.0, 1.0)))

    reversed_wav = _tone(300.0, SAMPLE_RATE * 2).flip(0)
    results.append(_verify("reversed_audio", reversed_wav))

    results.append(_verify("zero_input", _silence(SAMPLE_RATE * 2)))

    # Empty waveform
    try:
        accepted, sim = verifier.verify("sv_report_user", torch.zeros(0), SAMPLE_RATE, threshold=0.0)
        results.append({"scenario": "empty_waveform", "accepted": accepted, "similarity": round(sim, 4), "error": None})
    except (RuntimeError, ValueError) as exc:
        results.append({"scenario": "empty_waveform", "accepted": False, "similarity": 0.0, "error": str(exc)[:120]})

    results.append(_verify("unregistered_user", _tone(440.0, SAMPLE_RATE), user_id="no_such_user"))

    return results


# ===========================================================================
# SECTION 3 — Audit Log Evaluation
# ===========================================================================

def run_audit_log_evaluation() -> List[Dict[str, Any]]:
    from echoself_ai.audit.logger import AuditLogger, AuditRecord

    results: List[Dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp())

    # Test 1: log 5 records → read_all returns 5
    log1 = AuditLogger(log_path=str(tmp / "log1.jsonl"))
    for i in range(5):
        log1.log(AuditRecord(model_id="m", watermark_type="classical", action="synthesize", user_id=f"u{i}"))
    count = len(log1.read_all())
    results.append({"check": "5_records_read_all", "passed": count == 5, "detail": f"count={count}"})

    # Test 2: verify_integrity returns True
    ok = log1.verify_integrity()
    results.append({"check": "verify_integrity_clean", "passed": ok, "detail": f"integrity={ok}"})

    # Test 3: corrupt line → verify_integrity returns False
    log2_path = tmp / "log2.jsonl"
    log2 = AuditLogger(log_path=str(log2_path))
    log2.log(AuditRecord(model_id="m", watermark_type="classical", action="detect"))
    with open(log2_path, "a") as f:
        f.write("CORRUPTED_LINE\n")
    corrupted_ok = log2.verify_integrity()
    results.append({"check": "verify_integrity_corrupted", "passed": not corrupted_ok, "detail": f"integrity={corrupted_ok}"})

    # Test 4: event_ids are correct
    log3 = AuditLogger(log_path=str(tmp / "log3.jsonl"))
    ids_logged = []
    for i in range(5):
        rec = AuditRecord(model_id="m", watermark_type="classical", action="enroll", user_id=f"u{i}")
        log3.log(rec)
        ids_logged.append(rec.event_id)
    ids_read = [r.event_id for r in log3.read_all()]
    ids_match = ids_logged == ids_read
    results.append({"check": "event_ids_correct", "passed": ids_match, "detail": f"logged={ids_logged[:2]}..."})

    # Test 5: duplicate records → 2 occurrences
    log4 = AuditLogger(log_path=str(tmp / "log4.jsonl"))
    dup_rec = AuditRecord(model_id="m", watermark_type="classical", action="detect",
                          event_id="fixed-dup-id")
    log4.log(dup_rec)
    log4.log(dup_rec)
    dup_count = sum(1 for r in log4.read_all() if r.event_id == "fixed-dup-id")
    results.append({"check": "duplicate_append_twice", "passed": dup_count == 2, "detail": f"dup_count={dup_count}"})

    # Test 6: different users coexist
    log5 = AuditLogger(log_path=str(tmp / "log5.jsonl"))
    for user in ["alice", "bob", "carol"]:
        log5.log(AuditRecord(model_id="m", watermark_type="none", action="verify", user_id=user))
    read_users = {r.user_id for r in log5.read_all()}
    users_ok = read_users == {"alice", "bob", "carol"}
    results.append({"check": "different_users_coexist", "passed": users_ok, "detail": f"users={sorted(read_users)}"})

    return results


# ===========================================================================
# SECTION 4 — Safeguard Evaluation
# ===========================================================================

def run_safeguard_evaluation() -> List[Dict[str, Any]]:
    from echoself_ai.voice.consent import VoiceConsentManager
    from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
    from echoself_ai.pipeline import SpeechPipeline, PipelineConfig

    results: List[Dict[str, Any]] = []
    tmp = Path(tempfile.mkdtemp())

    # 1. Access without consent → PermissionError
    mgr = VoiceConsentManager(storage_dir=tmp / "c1")
    try:
        mgr.enforce_gate("u", "s", scope="synthesis")
        results.append({"check": "consent_gate_no_consent", "passed": False, "detail": "did not raise"})
    except PermissionError:
        results.append({"check": "consent_gate_no_consent", "passed": True, "detail": "raised PermissionError ✓"})

    # 2. Access after granting consent succeeds
    mgr2 = VoiceConsentManager(storage_dir=tmp / "c2")
    mgr2.register_consent("u", "s", scope="synthesis")
    try:
        mgr2.enforce_gate("u", "s", scope="synthesis")
        results.append({"check": "consent_gate_with_consent", "passed": True, "detail": "no exception ✓"})
    except PermissionError as e:
        results.append({"check": "consent_gate_with_consent", "passed": False, "detail": str(e)[:80]})

    # 3. Access after revoking → PermissionError
    mgr3 = VoiceConsentManager(storage_dir=tmp / "c3")
    mgr3.register_consent("u", "s", scope="synthesis")
    mgr3.revoke_consent("u", "s")
    try:
        mgr3.enforce_gate("u", "s", scope="synthesis")
        results.append({"check": "consent_gate_after_revoke", "passed": False, "detail": "did not raise after revoke"})
    except PermissionError:
        results.append({"check": "consent_gate_after_revoke", "passed": True, "detail": "raised PermissionError after revoke ✓"})

    # 4. AcousticModel + consent_manager + no user_id → PermissionError
    mgr4 = VoiceConsentManager(storage_dir=tmp / "c4")
    model = AcousticModel(AcousticModelConfig())
    model.eval()
    token_ids = torch.zeros(1, 5, dtype=torch.long)
    try:
        model(token_ids, user_id=None, raw_speaker_id=None, consent_manager=mgr4)
        results.append({"check": "acoustic_model_no_user_id", "passed": False, "detail": "did not raise"})
    except PermissionError:
        results.append({"check": "acoustic_model_no_user_id", "passed": True, "detail": "raised PermissionError ✓"})

    # 5. Pipeline with 10-sample input → error
    cfg = PipelineConfig(sample_rate=SAMPLE_RATE, audit_enabled=False)
    pipeline = SpeechPipeline(cfg)
    r = pipeline.process(torch.zeros(10, dtype=torch.float32), sample_rate=SAMPLE_RATE)
    results.append({
        "check": "pipeline_too_short_10_samples",
        "passed": not r.success and r.error is not None,
        "detail": f"success={r.success} error={r.error!r}"[:120],
    })

    # 6. Pipeline with NaN audio → error
    nan_audio = torch.full((SAMPLE_RATE,), float("nan"), dtype=torch.float32)
    r2 = pipeline.process(nan_audio, sample_rate=SAMPLE_RATE)
    results.append({
        "check": "pipeline_nan_audio",
        "passed": not r2.success and r2.error is not None,
        "detail": f"success={r2.success} error={r2.error!r}"[:120],
    })

    # 7. Pipeline with empty text_input → graceful
    valid_audio = _tone(440.0, SAMPLE_RATE * 2)
    r3 = pipeline.process(valid_audio, sample_rate=SAMPLE_RATE, text_input="")
    from echoself_ai.pipeline import PipelineResult
    graceful = isinstance(r3, PipelineResult)
    results.append({
        "check": "pipeline_empty_text_input",
        "passed": graceful,
        "detail": f"returned PipelineResult={graceful} transcript={r3.transcript!r}",
    })

    # 8. SpeakerVerifier unregistered user → (False, 0.0)
    from echoself_ai.voice.speaker_verification import SpeakerVerifier
    verifier = SpeakerVerifier(storage_dir=tmp / "sv")
    accepted, sim = verifier.verify("nobody", _tone(440.0, SAMPLE_RATE), SAMPLE_RATE)
    results.append({
        "check": "sv_unregistered_user",
        "passed": not accepted and sim == 0.0,
        "detail": f"accepted={accepted} sim={sim}",
    })

    return results


# ===========================================================================
# SECTION 5 — VAD Robustness
# ===========================================================================

def run_vad_robustness() -> List[Dict[str, Any]]:
    from echoself_ai.voice.vad import EnergyVAD
    from echoself_ai.voice.vad_config import VADConfig

    results: List[Dict[str, Any]] = []

    def make_vad(sr: int = SAMPLE_RATE) -> EnergyVAD:
        cfg = VADConfig(method="energy", energy_threshold=-4.0, sample_rate=sr,
                        min_speech_duration_ms=50, hangover_ms=200)
        return EnergyVAD(cfg)

    # 1. Pure silence → 0 segments
    vad = make_vad()
    segs = vad.detect(_silence(SAMPLE_RATE * 2), SAMPLE_RATE)
    results.append({"check": "pure_silence_zero_segments", "passed": segs == [], "detail": f"segments={segs}"})

    # 2. 100 ms audio → no crash
    try:
        segs = make_vad().detect(_tone(440.0, int(0.1 * SAMPLE_RATE)), SAMPLE_RATE)
        results.append({"check": "very_short_100ms_no_crash", "passed": isinstance(segs, list), "detail": f"segments={segs}"})
    except Exception as e:
        results.append({"check": "very_short_100ms_no_crash", "passed": False, "detail": str(e)[:80]})

    # 3. Very loud / clipped audio → no NaN
    loud = torch.clamp(_tone(440.0, SAMPLE_RATE * 2, amp=10.0), -1.0, 1.0)
    try:
        segs = make_vad().detect(loud, SAMPLE_RATE)
        has_nan = any(math.isnan(s) or math.isnan(e) for s, e in segs)
        results.append({"check": "very_loud_clipped_no_nan", "passed": not has_nan, "detail": f"segments={segs[:3]}"})
    except Exception as e:
        results.append({"check": "very_loud_clipped_no_nan", "passed": False, "detail": str(e)[:80]})

    # 4. All-NaN audio → raises or returns empty
    nan_audio = torch.full((SAMPLE_RATE,), float("nan"), dtype=torch.float32)
    try:
        segs = make_vad().detect(nan_audio, SAMPLE_RATE)
        results.append({"check": "all_nan_audio_no_silent_crash", "passed": isinstance(segs, list),
                         "detail": f"returned list len={len(segs)}"})
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        results.append({"check": "all_nan_audio_no_silent_crash", "passed": True,
                         "detail": f"raised {type(exc).__name__} (acceptable)"})

    # 5. Different sample rates
    for sr in [8000, 16000, 22050]:
        try:
            n = sr
            t = torch.arange(n, dtype=torch.float32) / sr
            tone = 0.5 * torch.sin(2.0 * math.pi * 440.0 * t)
            segs = make_vad(sr).detect(tone, sr)
            ok = isinstance(segs, list) and len(segs) >= 1
            results.append({"check": f"sample_rate_{sr}Hz", "passed": ok,
                             "detail": f"segments={segs}"})
        except Exception as e:
            results.append({"check": f"sample_rate_{sr}Hz", "passed": False, "detail": str(e)[:80]})

    return results


# ===========================================================================
# Main — collect all results, write JSON, print table
# ===========================================================================

def main():
    print("=" * 70)
    print("EchoSelf Week 11 — Robustness Report")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": {},
    }

    # -----------------------------------------------------------------------
    sections = [
        ("Section 1: Watermark Robustness",    "watermark_robustness",          run_watermark_robustness),
        ("Section 2: Speaker Verification",     "speaker_verification_robustness", run_speaker_verification_robustness),
        ("Section 3: Audit Log Evaluation",     "audit_log_evaluation",          run_audit_log_evaluation),
        ("Section 4: Safeguard Evaluation",     "safeguard_evaluation",          run_safeguard_evaluation),
        ("Section 5: VAD Robustness",           "vad_robustness",                run_vad_robustness),
    ]

    total_passed = 0
    total_checks = 0

    for title, key, fn in sections:
        print(f"\n{title}")
        print("-" * 60)

        try:
            results = fn()
        except Exception:
            err = traceback.format_exc()
            print(f"  ERROR running section: {err}")
            report["sections"][key] = {"error": err}
            continue

        report["sections"][key] = results

        # Determine column names
        # Watermark robustness uses transform/correlation/detected columns;
        # others use check/passed/detail
        if key == "watermark_robustness":
            print(f"  {'Transform':<40}  {'Correlation':>12}  {'Detected':>8}")
            print(f"  {'-'*40}  {'-'*12}  {'-'*8}")
            for r in results:
                corr = f"{r['correlation']:.6f}" if r["correlation"] is not None else "ERROR"
                det = str(r["detected"])
                print(f"  {r['transform']:<40}  {corr:>12}  {det:>8}")
        elif key == "speaker_verification_robustness":
            print(f"  {'Scenario':<40}  {'Similarity':>10}  {'Accepted':>8}")
            print(f"  {'-'*40}  {'-'*10}  {'-'*8}")
            for r in results:
                sim = f"{r['similarity']:.4f}" if r["similarity"] is not None else "ERR"
                acc = str(r["accepted"])
                print(f"  {r['scenario']:<40}  {sim:>10}  {acc:>8}")
        else:
            print(f"  {'Check':<45}  {'Pass':>5}  {'Detail'}")
            print(f"  {'-'*45}  {'-'*5}  {'-'*25}")
            for r in results:
                mark = "✓" if r.get("passed") else "✗"
                detail = str(r.get("detail", ""))[:55]
                print(f"  {r.get('check', ''):<45}  {mark:>5}  {detail}")

        # Count passes for sections with "passed" field
        section_results_with_pass = [r for r in results if "passed" in r]
        section_pass = sum(1 for r in section_results_with_pass if r.get("passed"))
        section_total = len(section_results_with_pass)
        total_passed += section_pass
        total_checks += section_total

        if section_total > 0:
            print(f"\n  Passed: {section_pass}/{section_total}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TOTAL CHECKS PASSED: {total_passed}/{total_checks}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Write JSON report
    # -----------------------------------------------------------------------
    report_path = EXPERIMENTS_DIR / "robustness_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")
    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
