"""Prompt-to-video and temporal sequence generation subsystem for EchoSelf."""
from echoself_ai.video.frames import (
    FrameSequence,
    VideoConfig,
    generate_synthetic_video,
)
from echoself_ai.video.temporal import (
    FrameInterpolator,
    FrameInterpolatorConfig,
    TemporalConvBlock,
    compute_motion_magnitude,
    compute_temporal_consistency_loss,
)

__all__ = [
    "FrameInterpolator",
    "FrameInterpolatorConfig",
    "FrameSequence",
    "TemporalConvBlock",
    "VideoConfig",
    "compute_motion_magnitude",
    "compute_temporal_consistency_loss",
    "generate_synthetic_video",
]

