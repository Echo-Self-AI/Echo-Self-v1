"""EchoSelf AI service extensions: portrait processing and talking-head video generation.

These handlers are registered into EchoSelfAIService at import time via
monkey-patching — this avoids modifying the original unified.py file.

Supported job types added:
- portrait_process   — process a portrait image → 512×512 PNG avatar
- portrait_load      — load a previously processed avatar
- talking_head_video — portrait + audio → real animated MP4
"""

from __future__ import annotations

import base64
import struct
import tempfile
from pathlib import Path
from typing import Any

from echoself_ai.service.unified import EchoSelfAIService, ProcessRequest
from echoself_ai.vision.portrait_processor import PortraitProcessor, PortraitProcessorConfig
from echoself_ai.video.talking_head import TalkingHeadGenerator, TalkingHeadConfig


# Shared instances (lazy-init per service instance via attribute)
_PORTRAIT_PROCESSOR: PortraitProcessor | None = None
_VIDEO_GENERATOR: TalkingHeadGenerator | None = None


def _get_portrait_processor() -> PortraitProcessor:
    global _PORTRAIT_PROCESSOR
    if _PORTRAIT_PROCESSOR is None:
        _PORTRAIT_PROCESSOR = PortraitProcessor(PortraitProcessorConfig(
            output_size=512,
            storage_dir="data/avatars",
        ))
    return _PORTRAIT_PROCESSOR


def _get_video_generator() -> TalkingHeadGenerator:
    global _VIDEO_GENERATOR
    if _VIDEO_GENERATOR is None:
        _VIDEO_GENERATOR = TalkingHeadGenerator(TalkingHeadConfig(
            fps=25,
            output_width=512,
            output_height=512,
        ))
    return _VIDEO_GENERATOR


# ------------------------------------------------------------------
# Handler: portrait_process
# ------------------------------------------------------------------

def _handle_portrait_process(self: EchoSelfAIService, req: ProcessRequest) -> dict[str, Any]:
    """Process a user portrait into a standardised 512×512 PNG avatar.

    Payload fields:
        image_base64 (str, required): Base64-encoded image bytes.
        avatar_id (str, optional): Identifier for this avatar. Defaults to "primary".
    """
    image_b64 = req.payload.get("image_base64")
    if not image_b64:
        raise ValueError("portrait_process requires image_base64 in payload")

    avatar_id = str(req.payload.get("avatar_id", "primary"))

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError(f"Invalid image_base64: {exc}") from exc

    processor = _get_portrait_processor()
    result = processor.process(image_bytes, user_id=req.user_id, avatar_id=avatar_id)

    return {
        "avatar_id": result.avatar_id,
        "original_size": list(result.original_size),
        "output_size": list(result.output_size),
        "image_base64": result.image_base64,
        "thumbnail_base64": result.thumbnail_base64,
        "processing_method": result.processing_method,
        "sha256": result.sha256,
        "created_at": result.created_at,
        "output_path": result.output_path,
    }


# ------------------------------------------------------------------
# Handler: portrait_load
# ------------------------------------------------------------------

def _handle_portrait_load(self: EchoSelfAIService, req: ProcessRequest) -> dict[str, Any]:
    """Load a previously processed avatar.

    Payload fields:
        avatar_id (str, optional): Defaults to "primary".
    """
    avatar_id = str(req.payload.get("avatar_id", "primary"))
    processor = _get_portrait_processor()
    result = processor.load(user_id=req.user_id, avatar_id=avatar_id)

    if result is None:
        return {
            "found": False,
            "avatar_id": avatar_id,
            "message": "No portrait found. Upload a portrait first.",
        }

    return {
        "found": True,
        "avatar_id": result.avatar_id,
        "output_size": list(result.output_size),
        "image_base64": result.image_base64,
        "thumbnail_base64": result.thumbnail_base64,
        "processing_method": result.processing_method,
        "sha256": result.sha256,
        "created_at": result.created_at,
    }


# ------------------------------------------------------------------
# Handler: talking_head_video
# ------------------------------------------------------------------

def _handle_talking_head_video(self: EchoSelfAIService, req: ProcessRequest) -> dict[str, Any]:
    """Generate a talking-head MP4 from a portrait and audio.

    Payload fields:
        portrait_base64 (str): Base64 PNG/JPEG portrait (from portrait_process).
        audio_base64 (str, optional): Base64 WAV audio. If omitted, synthesize from text.
        text (str, optional): Text to synthesize if no audio_base64.
        avatar_id (str, optional): Load portrait from storage if no portrait_base64.
        fps (int, optional): Defaults to 25.
        width (int, optional): Output width. Defaults to 512.
        height (int, optional): Output height. Defaults to 512.
    """
    # --- Resolve portrait ---
    portrait_b64 = req.payload.get("portrait_base64")
    avatar_id = str(req.payload.get("avatar_id", "primary"))

    if not portrait_b64:
        # Try loading from storage
        processor = _get_portrait_processor()
        stored = processor.load(user_id=req.user_id, avatar_id=avatar_id)
        if stored:
            portrait_b64 = stored.image_base64
        else:
            raise ValueError(
                "talking_head_video requires portrait_base64 or a stored portrait (upload via portrait_process first)"
            )

    try:
        portrait_bytes = base64.b64decode(portrait_b64)
    except Exception as exc:
        raise ValueError(f"Invalid portrait_base64: {exc}") from exc

    # --- Resolve audio ---
    audio_b64 = req.payload.get("audio_base64")
    if not audio_b64:
        text = str(req.payload.get("text", "EchoSelf — your voice and identity, echoed digitally."))
        # Synthesize audio using the existing voice handler
        voice_result = self._handle_voice_synthesis(ProcessRequest(
            job_id=req.job_id + "-voice",
            job_type="voice_synthesis",
            user_id=req.user_id,
            payload={"text": text, "speaker_id": req.payload.get("speaker_id")},
        ))
        audio_b64 = voice_result.get("audio_base64", "")

    if not audio_b64:
        raise ValueError("Could not obtain audio for talking_head_video")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        raise ValueError(f"Invalid audio_base64: {exc}") from exc

    # Validate WAV header
    if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF":
        raise ValueError("audio_base64 is not a valid WAV file")

    # --- Configure generator ---
    fps = int(req.payload.get("fps", 25))
    width = int(req.payload.get("width", 512))
    height = int(req.payload.get("height", 512))

    cfg = TalkingHeadConfig(fps=fps, output_width=width, output_height=height)
    generator = TalkingHeadGenerator(cfg)

    # --- Generate ---
    output_dir = Path("data/videos") / _safe_id(req.user_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = generator.generate(
        portrait_bytes=portrait_bytes,
        audio_wav_bytes=audio_bytes,
        output_dir=str(output_dir),
        job_id=f"{_safe_id(req.user_id)}-{req.job_id[-8:]}",
    )

    return {
        "video_path": result.video_path,
        "video_base64": result.video_base64,
        "audio_base64": result.audio_base64,
        "duration_sec": result.duration_sec,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "num_frames": result.num_frames,
        "generation_method": result.generation_method,
        "has_audio": result.has_audio,
        "limitation": (
            "Geometric portrait animation — mouth movement is driven by audio amplitude, "
            "not neural lip-sync. Labels this as 'portrait animation', not 'face generation'."
        ),
    }


def _safe_id(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in "-_")[:48] or "default"


# ------------------------------------------------------------------
# Registration — patch the service class at import time
# ------------------------------------------------------------------

def register_extensions() -> None:
    """Inject new handlers into EchoSelfAIService."""
    new_types = ["portrait_process", "portrait_load", "talking_head_video"]

    for jt in new_types:
        if jt not in EchoSelfAIService.SUPPORTED_JOB_TYPES:
            EchoSelfAIService.SUPPORTED_JOB_TYPES.append(jt)

    EchoSelfAIService._handle_portrait_process = _handle_portrait_process
    EchoSelfAIService._handle_portrait_load = _handle_portrait_load
    EchoSelfAIService._handle_talking_head_video = _handle_talking_head_video


# Auto-register when this module is imported
register_extensions()
