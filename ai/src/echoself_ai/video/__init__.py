"""Prompt-to-video and temporal sequence generation subsystem for EchoSelf."""
from echoself_ai.video.frames import (
    FrameSequence,
    VideoConfig,
    generate_synthetic_video,
)

__all__ = [
    "FrameSequence",
    "VideoConfig",
    "generate_synthetic_video",
]
