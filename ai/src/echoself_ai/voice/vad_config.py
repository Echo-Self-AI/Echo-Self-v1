"""VAD configuration dataclass for the EchoSelf voice activity detection module."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection.

    Attributes:
        frame_duration_ms: Duration of each analysis frame in milliseconds.
            Must be one of 10, 20, or 30 ms.
        energy_threshold: Log-energy threshold for energy-based VAD.
            Frames with log-energy above this value are considered speech.
        min_speech_duration_ms: Minimum speech segment duration to keep.
            Segments shorter than this are discarded (noise suppression).
        hangover_ms: How many milliseconds to extend a speech segment after
            the last active frame before declaring silence. Prevents clipping
            trailing consonants / sibilants.
        sample_rate: Expected sample rate in Hz. Used to validate inputs and
            compute frame sizes.
        method: VAD backend to use. 'energy' uses pure signal-energy heuristics;
            'neural' uses a tiny trained LSTM model.
    """

    frame_duration_ms: int = 20
    energy_threshold: float = -4.0
    min_speech_duration_ms: int = 100
    hangover_ms: int = 200
    sample_rate: int = 16000
    method: Literal["energy", "neural"] = "energy"

    def __post_init__(self) -> None:
        if self.frame_duration_ms not in (10, 20, 30):
            raise ValueError(
                f"frame_duration_ms must be 10, 20, or 30. Got {self.frame_duration_ms}"
            )
        if self.min_speech_duration_ms < 0:
            raise ValueError("min_speech_duration_ms must be non-negative")
        if self.hangover_ms < 0:
            raise ValueError("hangover_ms must be non-negative")
        if self.sample_rate < 8000:
            raise ValueError(f"sample_rate must be at least 8000. Got {self.sample_rate}")
        if self.method not in ("energy", "neural"):
            raise ValueError(f"method must be 'energy' or 'neural'. Got {self.method!r}")

    @property
    def frame_samples(self) -> int:
        """Number of waveform samples in one analysis frame."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def frame_duration_sec(self) -> float:
        """Frame duration in seconds."""
        return self.frame_duration_ms / 1000.0

    @property
    def min_speech_frames(self) -> int:
        """Minimum number of consecutive active frames to constitute speech."""
        return max(1, int(self.min_speech_duration_ms / self.frame_duration_ms))

    @property
    def hangover_frames(self) -> int:
        """Number of trailing frames to include after speech ends (hangover)."""
        return max(0, int(self.hangover_ms / self.frame_duration_ms))
