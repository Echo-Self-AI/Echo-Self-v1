"""Prompt-to-video and temporal sequence generation subsystem for EchoSelf."""
from echoself_ai.video.metrics import (
    compute_temporal_flicker_score,
    compute_video_psnr,
    compute_video_ssim,
    evaluate_video_pipeline,
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
    "compute_temporal_flicker_score",
    "compute_video_generation_loss",
    "compute_video_psnr",
    "compute_video_ssim",
    "evaluate_video_pipeline",
    "generate_synthetic_video",
]



