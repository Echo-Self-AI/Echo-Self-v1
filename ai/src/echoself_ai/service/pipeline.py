"""End-to-end multimodal orchestrator linking text, voice, avatar, and video with unified consent gating."""
from dataclasses import asdict, dataclass
import time
from typing import Any, Optional

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
