"""Prompt-to-video and temporal sequence generation subsystem for EchoSelf."""
from echoself_ai.video.frames import (
    FrameSequence,
    VideoConfig,
    generate_synthetic_video,
)
from echoself_ai.video.pipeline import (
    PromptTextEncoder,
    PromptToVideoConfig,
    PromptToVideoGenerator,
    compute_video_generation_loss,
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
    "PromptTextEncoder",
    "PromptToVideoConfig",
    "PromptToVideoGenerator",
    "TemporalConvBlock",
    "VideoConfig",
    "compute_motion_magnitude",
    "compute_temporal_consistency_loss",
    "compute_video_generation_loss",
    "generate_synthetic_video",
]


