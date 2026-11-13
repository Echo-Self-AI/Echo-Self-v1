"""
Configuration dataclass for classical spread-spectrum audio watermarking.
"""

from dataclasses import dataclass, field
import uuid


@dataclass
class ClassicalWatermarkConfig:
    """Configuration for classical spread-spectrum audio watermarking.

    Attributes:
        watermark_id: Unique identifier for the watermark (UUID string).
        strength: Embedding strength alpha — controls the trade-off between
                  imperceptibility and robustness.  Typical range: 0.001–0.1.
        n_bits: Number of bits used for the bit-error-rate calculation.
        seed: Global random seed used as a fallback / default.
        sample_rate: Expected sample rate of audio (Hz).
    """

    watermark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strength: float = 0.01
    n_bits: int = 32
    seed: int = 42
    sample_rate: int = 16000
