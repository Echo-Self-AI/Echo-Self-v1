"""End-to-end multimodal orchestrator linking text, voice, avatar, and video with unified consent gating.

Also exposes :func:`speech_pipeline_endpoint` as a thin service wrapper around
:class:`~echoself_ai.pipeline.SpeechPipeline`.
"""
from dataclasses import asdict, dataclass
import base64
import io
import struct
import time
import wave
from typing import Any, Optional

import torch

from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest, ProcessResult


class MultimodalConsentError(RuntimeError):
    """Raised when an end-to-end digital twin interaction fails consent validation."""


@dataclass
class DigitalTwinResponse:
    """Synchronized multimodal output from a digital twin persona."""

    user_id: str
    prompt: str
    text_reply: str
    audio_base64: str
    video_gif_base64: str
    audio_duration_seconds: float
    video_num_frames: int
    consent_audit: dict[str, Any]
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalPipeline:
    """Coordinates complex cross-modal workflows: text reasoning -> speech -> animated video."""

    def __init__(self, service: EchoSelfAIService) -> None:
        self.service = service

    def execute_interaction(
        self,
        user_id: str,
        prompt: str,
        avatar_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        max_new_tokens: int = 15,
        num_video_frames: int = 8,
        video_fps: int = 12,
    ) -> DigitalTwinResponse:
        """Execute full digital twin interaction generating text, voice audio, and animated video."""
        t0 = time.perf_counter()

        # 1. Audit Consent
        consent_audit: dict[str, Any] = {
            "user_id": user_id,
            "avatar_id": avatar_id,
            "speaker_id": speaker_id,
        }

        if speaker_id:
            has_voice = self.service.voice_consent.verify_consent(
                user_id, speaker_id, scope="synthesis"
            ) or self.service.voice_consent.verify_consent(user_id, speaker_id, scope="all")
            consent_audit["voice_authorized"] = has_voice
            if not has_voice:
                raise MultimodalConsentError(
                    f"Multimodal interaction blocked: active voice consent not found for "
                    f"user='{user_id}', speaker='{speaker_id}'."
                )
        else:
            consent_audit["voice_authorized"] = True

        if avatar_id:
            has_visual = self.service.visual_consent.verify_consent(
                user_id, avatar_id, scope="generation"
            ) or self.service.visual_consent.verify_consent(user_id, avatar_id, scope="all")
            consent_audit["visual_authorized"] = has_visual
            if not has_visual:
                raise MultimodalConsentError(
                    f"Multimodal interaction blocked: active visual consent not found for "
                    f"user='{user_id}', avatar='{avatar_id}'."
                )
        else:
            consent_audit["visual_authorized"] = True

        # 2. Step 1: Text Generation (LM)
        req_text = ProcessRequest(
            job_id=f"twin-text-{int(t0)}",
            job_type="llm_generate",
            user_id=user_id,
            payload={"prompt": prompt, "max_new_tokens": max_new_tokens},
        )
        res_text = self.service.process(req_text)
        if res_text.status != "success":
            raise RuntimeError(f"Text generation failed: {res_text.error}")
        text_reply = res_text.output.get("generated_text", prompt)

        # 3. Step 2: Speech Synthesis (Voice AI)
        req_voice = ProcessRequest(
            job_id=f"twin-voice-{int(t0)}",
            job_type="voice_synthesis",
            user_id=user_id,
            payload={"text": text_reply, "speaker_id": speaker_id},
        )
        res_voice = self.service.process(req_voice)
        if res_voice.status != "success":
            raise RuntimeError(f"Voice synthesis failed: {res_voice.error}")
        audio_b64 = res_voice.output.get("audio_base64", "")
        duration_s = res_voice.output.get("duration_seconds", 0.0)

        # 4. Step 3: Video Animation (Prompt-to-Video)
        req_video = ProcessRequest(
            job_id=f"twin-video-{int(t0)}",
            job_type="video_generate",
            user_id=user_id,
            payload={
                "prompt": f"avatar saying {prompt[:20]}",
                "avatar_id": avatar_id,
                "num_frames": num_video_frames,
                "fps": video_fps,
            },
        )
        res_video = self.service.process(req_video)
        if res_video.status != "success":
            raise RuntimeError(f"Video animation failed: {res_video.error}")
        video_b64 = res_video.output.get("video_gif_base64", "")

        total_latency = (time.perf_counter() - t0) * 1000.0

        return DigitalTwinResponse(
            user_id=user_id,
            prompt=prompt,
            text_reply=text_reply,
            audio_base64=audio_b64,
            video_gif_base64=video_b64,
            audio_duration_seconds=duration_s,
            video_num_frames=num_video_frames,
            consent_audit=consent_audit,
            execution_time_ms=total_latency,
        )


# ---------------------------------------------------------------------------
# speech_pipeline endpoint
# ---------------------------------------------------------------------------

def _tensor_to_b64_wav(waveform: torch.Tensor, sample_rate: int) -> str:
    """Encode a waveform tensor as a base64-encoded WAV byte string."""
    flat = waveform.detach().cpu().float().squeeze()
    clamped = torch.clamp(flat, -1.0, 1.0)
    ints = (clamped * 32767.0).to(torch.int16).tolist()
    raw_bytes = struct.pack(f"<{len(ints)}h", *ints)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def speech_pipeline_endpoint(
    input_waveform: torch.Tensor,
    sample_rate: int = 16000,
    user_id: Optional[str] = None,
    text_input: Optional[str] = None,
    pipeline_config: Optional[Any] = None,
) -> dict[str, Any]:
    """Service endpoint that wraps :class:`~echoself_ai.pipeline.SpeechPipeline`.

    Accepts a raw waveform (or text_input) and returns a JSON-serialisable
    dict containing the pipeline result — including base64-encoded WAV audio
    when output audio was produced.

    Args:
        input_waveform: 1-D float32 waveform tensor.
        sample_rate: Sample rate of *input_waveform* in Hz.
        user_id: Optional user identifier forwarded to the audit log.
        text_input: When provided, skip ASR and synthesise this text directly.
        pipeline_config: Optional :class:`~echoself_ai.pipeline.PipelineConfig`.
            Defaults to a standard config when None.

    Returns:
        Dict with keys: ``job_id``, ``transcript``, ``audio_base64``,
        ``audio_sample_rate``, ``watermark_id``, ``detection_result``,
        ``audit_event_id``, ``stage_latencies``, ``total_latency_ms``,
        ``real_time_factor``, ``success``, ``error``.
    """
    from echoself_ai.pipeline import SpeechPipeline, PipelineConfig

    cfg = pipeline_config if pipeline_config is not None else PipelineConfig(sample_rate=sample_rate)
    pipeline = SpeechPipeline(config=cfg)
    result = pipeline.process(
        input_waveform,
        sample_rate=sample_rate,
        user_id=user_id,
        text_input=text_input,
    )

    audio_b64 = ""
    if result.output_waveform is not None and result.output_sample_rate is not None:
        try:
            audio_b64 = _tensor_to_b64_wav(result.output_waveform, result.output_sample_rate)
        except Exception:
            audio_b64 = ""

    return {
        "job_id": result.job_id,
        "transcript": result.transcript,
        "audio_base64": audio_b64,
        "audio_sample_rate": result.output_sample_rate,
        "watermark_id": result.watermark_id,
        "detection_result": result.detection_result,
        "audit_event_id": result.audit_event_id,
        "stage_latencies": result.stage_latencies,
        "total_latency_ms": result.total_latency_ms,
        "real_time_factor": result.real_time_factor,
        "success": result.success,
        "error": result.error,
    }
