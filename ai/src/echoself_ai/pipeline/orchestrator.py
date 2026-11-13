"""End-to-end speech pipeline orchestrator for EchoSelf.

Chains VAD → ASR → NVS (TTS) → watermark embed → watermark detect → audit log
into a single :class:`SpeechPipeline` with per-stage latency measurement.

Usage::

    from echoself_ai.pipeline import SpeechPipeline, PipelineConfig
    from echoself_ai.voice.audio import generate_synthetic_tone

    cfg = PipelineConfig(sample_rate=16000)
    pipeline = SpeechPipeline(cfg)

    tone = generate_synthetic_tone(freq_hz=440.0, duration_seconds=2.0, sample_rate=16000)
    result = pipeline.process(tone, sample_rate=16000)
    print(result.total_latency_ms, result.real_time_factor)

    profile = pipeline.run_latency_profile(n_trials=5)
    print(profile["mean"])
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for the end-to-end speech pipeline.

    Attributes:
        sample_rate: Expected input audio sample rate in Hz.
        vad_enabled: Run voice activity detection to segment speech.
        asr_enabled: Run automatic speech recognition to transcribe.
        nvs_enabled: Run neural voice synthesis (TTS) to produce output audio.
        watermark_enabled: Embed and verify a spread-spectrum watermark.
        audit_enabled: Append an audit record to the append-only log.
        timeout_seconds: Maximum wall-clock time allowed for one ``process()`` call.
            Stages that exceed this budget are skipped gracefully.
        watermark_strength: Embedding alpha passed to :class:`SpreadSpectrumWatermarker`.
        speaker_verification_threshold: Cosine similarity threshold for
            :class:`SpeakerVerifier`.
    """

    sample_rate: int = 16000
    vad_enabled: bool = True
    asr_enabled: bool = True
    nvs_enabled: bool = True
    watermark_enabled: bool = True
    audit_enabled: bool = True
    timeout_seconds: float = 30.0
    watermark_strength: float = 0.01
    speaker_verification_threshold: float = 0.75


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result returned by :meth:`SpeechPipeline.process`.

    Attributes:
        job_id: Unique identifier for this pipeline run (UUID string).
        input_duration_seconds: Duration of the input waveform in seconds.
        transcript: Transcribed text from ASR (or None when ASR is disabled / failed).
        output_waveform: Synthesised (and watermarked) waveform tensor, or None.
        output_sample_rate: Sample rate of the output waveform, or None.
        watermark_id: UUID of the embedded watermark, or None.
        detection_result: True when the watermark was successfully verified, None otherwise.
        audit_event_id: UUID of the audit log record, or None when audit is disabled.
        stage_latencies: Mapping of stage name → wall-clock latency in milliseconds.
        total_latency_ms: Sum of all stage latencies.
        real_time_factor: ``total_latency_ms / (input_duration_seconds * 1000)``.
            Values < 1 mean faster-than-real-time processing.
        success: True when the pipeline completed without a fatal error.
        error: Description of the first error encountered, or None.
    """

    job_id: str
    input_duration_seconds: float
    transcript: Optional[str]
    output_waveform: Optional[torch.Tensor]
    output_sample_rate: Optional[int]
    watermark_id: Optional[str]
    detection_result: Optional[bool]
    audit_event_id: Optional[str]
    stage_latencies: Dict[str, float]
    total_latency_ms: float
    real_time_factor: float
    success: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# SpeechPipeline
# ---------------------------------------------------------------------------

class SpeechPipeline:
    """End-to-end speech pipeline with per-stage latency profiling.

    Initialises all sub-modules once at construction time, then processes
    audio through up to seven stages each time :meth:`process` is called.

    Args:
        config: :class:`PipelineConfig` controlling which stages are active.

    Stages (all individually guarded by try/except):
        1. ``validate``  — Check for NaN, minimum length.
        2. ``vad``       — Voice Activity Detection; extract speech segments.
        3. ``asr``       — Automatic Speech Recognition; transcribe first segment.
        4. ``nvs``       — Neural Voice Synthesis; synthesise output from text.
        5. ``watermark`` — Embed spread-spectrum watermark into output audio.
        6. ``detect``    — Verify the watermark was embedded correctly.
        7. ``audit``     — Write an append-only audit log record.
    """

    # Minimum waveform length (in samples) below which we reject input.
    _MIN_SAMPLES: int = 160  # 10 ms at 16 kHz

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._init_submodules()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_submodules(self) -> None:
        """Instantiate all sub-modules.  Errors here propagate to the caller."""
        from echoself_ai.voice.vad import create_vad
        from echoself_ai.voice.asr import SpeechRecognizer
        from echoself_ai.voice.nvs import TextToSpeech, NVSConfig
        from echoself_ai.watermark.classical import SpreadSpectrumWatermarker
        from echoself_ai.watermark.config import ClassicalWatermarkConfig
        from echoself_ai.watermark.detector import WatermarkDetector
        from echoself_ai.audit.logger import AuditLogger

        cfg = self.config

        # VAD
        self._vad = create_vad(
            method="energy",
            sample_rate=cfg.sample_rate,
            min_speech_duration_ms=100,
            hangover_ms=200,
        )

        # ASR
        self._asr = SpeechRecognizer()

        # NVS (TTS)
        nvs_cfg = NVSConfig(sample_rate=cfg.sample_rate)
        self._tts = TextToSpeech(config=nvs_cfg)

        # Classical watermarker (placeholder config — real watermark_id is
        # generated per job in process())
        wm_cfg = ClassicalWatermarkConfig(
            watermark_id="placeholder",
            strength=cfg.watermark_strength,
            sample_rate=cfg.sample_rate,
        )
        self._watermarker = SpreadSpectrumWatermarker(wm_cfg)

        # Watermark detector
        self._detector = WatermarkDetector()

        # Audit logger
        self._audit_logger = AuditLogger()

        logger.info(
            "SpeechPipeline initialised (vad=%s asr=%s nvs=%s wm=%s audit=%s)",
            cfg.vad_enabled,
            cfg.asr_enabled,
            cfg.nvs_enabled,
            cfg.watermark_enabled,
            cfg.audit_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_waveform: torch.Tensor,
        sample_rate: int,
        user_id: Optional[str] = None,
        text_input: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full pipeline on *input_waveform*.

        Args:
            input_waveform: 1-D float32 waveform tensor.
            sample_rate: Sample rate of *input_waveform* in Hz.
            user_id: Optional user identifier forwarded to the audit log.
            text_input: When provided, skip ASR and synthesise this text directly.

        Returns:
            :class:`PipelineResult` with latency measurements and all outputs.
        """
        job_id = str(uuid.uuid4())
        stage_latencies: Dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # Flatten / squeeze to 1-D
        waveform = input_waveform.detach().cpu().float().squeeze()
        if waveform.dim() == 0:
            waveform = waveform.unsqueeze(0)

        input_duration_seconds = waveform.numel() / max(sample_rate, 1)

        # Mutable state shared across stages
        transcript: Optional[str] = None
        output_waveform: Optional[torch.Tensor] = None
        output_sample_rate: Optional[int] = None
        watermark_id: Optional[str] = None
        detection_result: Optional[bool] = None
        audit_event_id: Optional[str] = None
        first_error: Optional[str] = None
        success = True

        # Helpers -----------------------------------------------------------

        def _record(stage: str, t0: float) -> None:
            stage_latencies[stage] = (time.perf_counter() - t0) * 1000.0

        def _timeout_exceeded() -> bool:
            elapsed = time.perf_counter() - pipeline_start
            return elapsed > self.config.timeout_seconds

        # -------------------------------------------------------------------
        # Stage 1: Validate input
        # -------------------------------------------------------------------
        t = time.perf_counter()
        try:
            if waveform.numel() < self._MIN_SAMPLES:
                raise ValueError(
                    f"Input too short: {waveform.numel()} samples "
                    f"(minimum {self._MIN_SAMPLES})"
                )
            if not torch.isfinite(waveform).all():
                raise ValueError("Input waveform contains NaN or Inf values")
        except Exception as exc:
            first_error = f"validate: {exc}"
            success = False
            logger.warning("Pipeline validate stage failed: %s", exc)
        finally:
            _record("validate", t)

        if not success:
            return self._build_result(
                job_id=job_id,
                input_duration_seconds=input_duration_seconds,
                transcript=transcript,
                output_waveform=output_waveform,
                output_sample_rate=output_sample_rate,
                watermark_id=watermark_id,
                detection_result=detection_result,
                audit_event_id=audit_event_id,
                stage_latencies=stage_latencies,
                pipeline_start=pipeline_start,
                success=success,
                error=first_error,
            )

        # -------------------------------------------------------------------
        # Stage 2: VAD
        # -------------------------------------------------------------------
        t = time.perf_counter()
        speech_segments: List[torch.Tensor] = []
        if self.config.vad_enabled and not _timeout_exceeded():
            try:
                speech_segments = self._vad.segment_waveform(waveform, sample_rate)
                logger.debug("VAD found %d segment(s)", len(speech_segments))
            except Exception as exc:
                if first_error is None:
                    first_error = f"vad: {exc}"
                logger.warning("Pipeline VAD stage failed: %s", exc)
        _record("vad", t)

        # -------------------------------------------------------------------
        # Stage 3: ASR
        # -------------------------------------------------------------------
        t = time.perf_counter()
        if text_input is not None:
            # Caller provided text — skip ASR entirely
            transcript = text_input
        elif self.config.asr_enabled and not _timeout_exceeded():
            try:
                # Use first speech segment if VAD found any, else use full waveform
                asr_input = speech_segments[0] if speech_segments else waveform
                transcript = self._asr.recognize(asr_input, sample_rate)
                logger.debug("ASR transcript: %r", transcript)
            except Exception as exc:
                if first_error is None:
                    first_error = f"asr: {exc}"
                logger.warning("Pipeline ASR stage failed: %s", exc)
        _record("asr", t)

        # -------------------------------------------------------------------
        # Stage 4: NVS (TTS)
        # -------------------------------------------------------------------
        t = time.perf_counter()
        if self.config.nvs_enabled and not _timeout_exceeded():
            tts_text = transcript or text_input or ""
            if tts_text.strip():
                try:
                    wav, sr = self._tts.synthesize(tts_text)
                    output_waveform = wav
                    output_sample_rate = sr
                    logger.debug("NVS produced %d samples @ %d Hz", len(wav), sr)
                except Exception as exc:
                    if first_error is None:
                        first_error = f"nvs: {exc}"
                    logger.warning("Pipeline NVS stage failed: %s", exc)
            else:
                # No text to synthesise — produce silence as a placeholder
                # so downstream stages have something to work with
                output_waveform = torch.zeros(self.config.sample_rate, dtype=torch.float32)
                output_sample_rate = self.config.sample_rate
        _record("nvs", t)

        # -------------------------------------------------------------------
        # Stage 5: Watermark embed
        # -------------------------------------------------------------------
        t = time.perf_counter()
        if self.config.watermark_enabled and output_waveform is not None and not _timeout_exceeded():
            try:
                from echoself_ai.watermark.config import ClassicalWatermarkConfig
                from echoself_ai.watermark.classical import SpreadSpectrumWatermarker

                watermark_id = str(uuid.uuid4())
                wm_cfg = ClassicalWatermarkConfig(
                    watermark_id=watermark_id,
                    strength=self.config.watermark_strength,
                    sample_rate=output_sample_rate or self.config.sample_rate,
                )
                wm = SpreadSpectrumWatermarker(wm_cfg)
                output_waveform = wm.embed(output_waveform, watermark_id)
                logger.debug("Watermark embedded: %s", watermark_id)
            except Exception as exc:
                if first_error is None:
                    first_error = f"watermark: {exc}"
                logger.warning("Pipeline watermark embed stage failed: %s", exc)
        _record("watermark", t)

        # -------------------------------------------------------------------
        # Stage 6: Watermark detect (verify)
        # -------------------------------------------------------------------
        t = time.perf_counter()
        if (
            self.config.watermark_enabled
            and output_waveform is not None
            and watermark_id is not None
            and not _timeout_exceeded()
        ):
            try:
                result = self._detector.detect_classical(
                    output_waveform,
                    watermark_id=watermark_id,
                    sample_rate=output_sample_rate or self.config.sample_rate,
                )
                detection_result = result.detected
                logger.debug(
                    "Watermark detection: detected=%s confidence=%.4f",
                    result.detected,
                    result.confidence,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = f"detect: {exc}"
                logger.warning("Pipeline watermark detect stage failed: %s", exc)
        _record("detect", t)

        # -------------------------------------------------------------------
        # Stage 7: Audit log
        # -------------------------------------------------------------------
        t = time.perf_counter()
        if self.config.audit_enabled and not _timeout_exceeded():
            try:
                from echoself_ai.audit.logger import AuditRecord, AuditLogger, compute_audio_hash

                # Compute a hash of the output audio (or input if no output)
                hash_waveform = output_waveform if output_waveform is not None else waveform
                audio_hash = compute_audio_hash(
                    hash_waveform,
                    output_sample_rate or sample_rate,
                )

                record = AuditRecord(
                    model_id="speech-pipeline-v1",
                    watermark_type="classical" if watermark_id else "none",
                    action="synthesize",
                    user_id=user_id,
                    watermark_id=watermark_id,
                    output_hash=audio_hash,
                    detection_result=detection_result,
                    metadata={
                        "job_id": job_id,
                        "input_duration_seconds": round(input_duration_seconds, 4),
                        "transcript": transcript or "",
                        "success": success and first_error is None,
                    },
                )
                self._audit_logger.log(record)
                audit_event_id = record.event_id
                logger.debug("Audit record written: %s", audit_event_id)
            except Exception as exc:
                if first_error is None:
                    first_error = f"audit: {exc}"
                logger.warning("Pipeline audit stage failed: %s", exc)
        _record("audit", t)

        # -------------------------------------------------------------------
        # Build final result
        # -------------------------------------------------------------------
        # Treat the pipeline as successful if no fatal error (validate) occurred;
        # individual stage failures are non-fatal but set error field.
        final_success = first_error is None

        return self._build_result(
            job_id=job_id,
            input_duration_seconds=input_duration_seconds,
            transcript=transcript,
            output_waveform=output_waveform,
            output_sample_rate=output_sample_rate,
            watermark_id=watermark_id,
            detection_result=detection_result,
            audit_event_id=audit_event_id,
            stage_latencies=stage_latencies,
            pipeline_start=pipeline_start,
            success=final_success,
            error=first_error,
        )

    # ------------------------------------------------------------------
    # Latency profiling
    # ------------------------------------------------------------------

    def run_latency_profile(self, n_trials: int = 5) -> Dict[str, Any]:
        """Run the pipeline *n_trials* times on 1-second synthetic audio and
        collect per-stage latency statistics.

        Args:
            n_trials: Number of trials to run (default 5).

        Returns:
            Dict with keys:

            * ``'mean'``  — {stage: mean ms}
            * ``'std'``   — {stage: std-dev ms}
            * ``'min'``   — {stage: min ms}
            * ``'max'``   — {stage: max ms}
            * ``'rtf_mean'``    — mean real-time factor across trials
            * ``'rtf_std'``     — std-dev of real-time factor
            * ``'total_mean_ms'`` — mean total latency across trials
            * ``'n_trials'``    — number of trials executed
        """
        from echoself_ai.voice.audio import generate_synthetic_tone

        all_latencies: List[Dict[str, float]] = []
        all_rtf: List[float] = []
        all_total: List[float] = []

        # Fixed synthetic stimulus for reproducibility
        tone = generate_synthetic_tone(
            freq_hz=440.0,
            duration_seconds=1.0,
            sample_rate=self.config.sample_rate,
        )

        for trial_idx in range(n_trials):
            try:
                result = self.process(
                    tone,
                    sample_rate=self.config.sample_rate,
                    text_input="hello world",
                )
                all_latencies.append(result.stage_latencies)
                all_rtf.append(result.real_time_factor)
                all_total.append(result.total_latency_ms)
            except Exception as exc:
                logger.warning("Latency profile trial %d failed: %s", trial_idx, exc)

        if not all_latencies:
            return {
                "mean": {},
                "std": {},
                "min": {},
                "max": {},
                "rtf_mean": 0.0,
                "rtf_std": 0.0,
                "total_mean_ms": 0.0,
                "n_trials": 0,
            }

        # Gather all stage names across all trials
        all_stages = set()
        for lat in all_latencies:
            all_stages.update(lat.keys())

        mean_map: Dict[str, float] = {}
        std_map: Dict[str, float] = {}
        min_map: Dict[str, float] = {}
        max_map: Dict[str, float] = {}

        import statistics

        for stage in sorted(all_stages):
            vals = [lat.get(stage, 0.0) for lat in all_latencies]
            mean_map[stage] = statistics.mean(vals)
            std_map[stage] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            min_map[stage] = min(vals)
            max_map[stage] = max(vals)

        return {
            "mean": mean_map,
            "std": std_map,
            "min": min_map,
            "max": max_map,
            "rtf_mean": statistics.mean(all_rtf) if all_rtf else 0.0,
            "rtf_std": statistics.stdev(all_rtf) if len(all_rtf) > 1 else 0.0,
            "total_mean_ms": statistics.mean(all_total) if all_total else 0.0,
            "n_trials": len(all_latencies),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        job_id: str,
        input_duration_seconds: float,
        transcript: Optional[str],
        output_waveform: Optional[torch.Tensor],
        output_sample_rate: Optional[int],
        watermark_id: Optional[str],
        detection_result: Optional[bool],
        audit_event_id: Optional[str],
        stage_latencies: Dict[str, float],
        pipeline_start: float,
        success: bool,
        error: Optional[str],
    ) -> PipelineResult:
        total_latency_ms = sum(stage_latencies.values())
        input_ms = input_duration_seconds * 1000.0
        real_time_factor = total_latency_ms / input_ms if input_ms > 0 else 0.0

        return PipelineResult(
            job_id=job_id,
            input_duration_seconds=input_duration_seconds,
            transcript=transcript,
            output_waveform=output_waveform,
            output_sample_rate=output_sample_rate,
            watermark_id=watermark_id,
            detection_result=detection_result,
            audit_event_id=audit_event_id,
            stage_latencies=stage_latencies,
            total_latency_ms=total_latency_ms,
            real_time_factor=real_time_factor,
            success=success,
            error=error,
        )
