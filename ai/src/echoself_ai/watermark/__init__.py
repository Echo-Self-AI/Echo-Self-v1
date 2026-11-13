"""
echoself_ai.watermark
=====================
Classical spread-spectrum audio watermarking.

Public API
----------
* :class:`ClassicalWatermarkConfig` — configuration dataclass
* :class:`SpreadSpectrumWatermarker` — embed / extract watermarks
* :func:`n_bit_watermark_id_to_bits` — hash a watermark ID to bits
* :func:`bit_error_rate` — compare two bit sequences
"""

from .config import ClassicalWatermarkConfig
from .classical import (
    SpreadSpectrumWatermarker,
    bit_error_rate,
    n_bit_watermark_id_to_bits,
)

__all__ = [
    "ClassicalWatermarkConfig",
    "SpreadSpectrumWatermarker",
    "bit_error_rate",
    "n_bit_watermark_id_to_bits",
]
