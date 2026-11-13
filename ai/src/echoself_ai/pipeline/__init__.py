"""EchoSelf speech pipeline package.

Provides end-to-end speech processing with latency profiling.

Public API
----------
* :class:`PipelineConfig` — configuration dataclass
* :class:`PipelineResult` — result dataclass returned by :meth:`SpeechPipeline.process`
* :class:`SpeechPipeline` — orchestrates VAD → ASR → NVS → watermark → audit
"""

from echoself_ai.pipeline.orchestrator import (
    PipelineConfig,
    PipelineResult,
    SpeechPipeline,
)

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SpeechPipeline",
]
