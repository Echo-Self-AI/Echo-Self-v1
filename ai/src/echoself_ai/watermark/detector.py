"""
Watermark detector: unified detection interface for classical and learned watermarks.

Provides:
- DetectionResult dataclass
- WatermarkDetector class with detect_classical, detect_learned, detect_all, detect_from_file
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of a single watermark detection attempt.

    Attributes:
        watermark_type: One of 'classical', 'learned', 'none', 'unsupported'.
        detected: True when a watermark was found above the detection threshold.
        confidence: Detection confidence in [0.0, 1.0].
        watermark_id: Identifier of the watermark searched for (classical only).
        correlation: Normalised cross-correlation (classical only).
        bit_accuracy: Fraction of bits correctly recovered (learned only).
        processing_time_ms: Wall-clock time taken for this detection call in ms.
        error: Set to a non-None string when an error occurred during detection.
    """

    watermark_type: str
    detected: bool
    confidence: float
    watermark_id: Optional[str] = None
    correlation: Optional[float] = None
    bit_accuracy: Optional[float] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# WatermarkDetector
# ---------------------------------------------------------------------------

class WatermarkDetector:
    """Unified watermark detector supporting classical and learned methods.

    Args:
        classical_threshold: Override the classical DETECTION_THRESHOLD.
                             Defaults to :attr:`SpreadSpectrumWatermarker.DETECTION_THRESHOLD`.
        checkpoint_dir: Directory containing learned watermark checkpoints
                        (``encoder.pt`` / ``decoder.pt``).  When *None* the
                        default location inside the repository is used.
    """

    def __init__(
        self,
        classical_threshold: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self._classical_threshold = classical_threshold
        self._checkpoint_dir = checkpoint_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_classical_watermarker(self, watermark_id: str, sample_rate: int):
        """Instantiate a SpreadSpectrumWatermarker for the given ID."""
        from .classical import SpreadSpectrumWatermarker
        from .config import ClassicalWatermarkConfig

        cfg = ClassicalWatermarkConfig(
            watermark_id=watermark_id,
            sample_rate=sample_rate,
        )
        wm = SpreadSpectrumWatermarker(cfg)
        if self._classical_threshold is not None:
            # Allow tests to override the threshold
            wm.DETECTION_THRESHOLD = self._classical_threshold  # type: ignore[assignment]
        return wm

    def _resolve_checkpoint_dir(self) -> Path:
        """Return the checkpoint directory path."""
        if self._checkpoint_dir is not None:
            return Path(self._checkpoint_dir)
        # detector.py lives at ai/src/echoself_ai/watermark/detector.py
        # Navigate up 4 levels: watermark → echoself_ai → src → ai
        this_file = Path(__file__).resolve()
        ai_dir = this_file.parents[3]
        return ai_dir / "checkpoints" / "learned_watermark"

    def _checkpoint_exists(self) -> bool:
        """Return True when both encoder.pt and decoder.pt are present."""
        cp_dir = self._resolve_checkpoint_dir()
        return (cp_dir / "encoder.pt").is_file() and (cp_dir / "decoder.pt").is_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_classical(
        self,
        waveform: torch.Tensor,
        watermark_id: str,
        sample_rate: int = 16000,
    ) -> DetectionResult:
        """Detect a classical spread-spectrum watermark in *waveform*.

        Args:
            waveform: 1-D float32 audio tensor.
            watermark_id: Identifier of the watermark to look for.
            sample_rate: Audio sample rate in Hz (informational, not used by
                         the detection algorithm itself).

        Returns:
            :class:`DetectionResult` with ``watermark_type='classical'``.
        """
        t0 = time.perf_counter()
        try:
            waveform = waveform.float().flatten()
            if waveform.numel() == 0:
                raise ValueError("Waveform is empty (zero samples)")

            wm = self._make_classical_watermarker(watermark_id, sample_rate)
            detected, correlation, ber = wm.extract(waveform, watermark_id)

            # Bit accuracy = 1 - BER
            bit_acc = 1.0 - float(ber)

            # Confidence: map correlation from [-1, 1] to [0, 1] using sigmoid-like
            # approach.  When correlation is well above the threshold the confidence
            # approaches 1; near zero it is ~0.5; below threshold it drops.
            # We use a simple affine mapping clamped to [0, 1]:
            #   confidence = (correlation + 1) / 2
            # but clamped so it reflects the detection decision.
            if detected:
                # Range the confidence between 0.5 and 1.0 based on how far above threshold
                threshold = wm.DETECTION_THRESHOLD
                excess = max(0.0, float(correlation) - threshold)
                confidence = min(1.0, 0.5 + excess / (abs(threshold) + 1e-6) * 0.5)
                confidence = max(0.5, confidence)
            else:
                # Below threshold → low confidence
                raw = (float(correlation) + 1.0) / 2.0
                confidence = min(0.49, raw)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                watermark_type="classical",
                detected=bool(detected),
                confidence=float(confidence),
                watermark_id=watermark_id,
                correlation=float(correlation),
                bit_accuracy=float(bit_acc),
                processing_time_ms=elapsed_ms,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                watermark_type="classical",
                detected=False,
                confidence=0.0,
                watermark_id=watermark_id,
                processing_time_ms=elapsed_ms,
                error=str(exc),
            )

    def detect_learned(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> DetectionResult:
        """Detect a learned watermark in *waveform*.

        Returns an 'unsupported' result when no trained checkpoint is found.

        Args:
            waveform: 1-D float32 audio tensor.
            sample_rate: Audio sample rate in Hz.

        Returns:
            :class:`DetectionResult` with ``watermark_type='learned'``
            or ``watermark_type='unsupported'``.
        """
        t0 = time.perf_counter()
        try:
            if not self._checkpoint_exists():
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return DetectionResult(
                    watermark_type="unsupported",
                    detected=False,
                    confidence=0.0,
                    processing_time_ms=elapsed_ms,
                    error="No trained checkpoint found; run LearnedWatermarker.train_model() first",
                )

            waveform = waveform.float().flatten()
            if waveform.numel() == 0:
                raise ValueError("Waveform is empty (zero samples)")

            from .learned import LearnedWatermarker, LearnedWatermarkConfig

            cp_dir = self._resolve_checkpoint_dir()
            cfg = LearnedWatermarkConfig(sample_rate=sample_rate)
            learned_wm = LearnedWatermarker(config=cfg, checkpoint_dir=str(cp_dir))
            learned_wm.load_model(
                encoder_path=str(cp_dir / "encoder.pt"),
                decoder_path=str(cp_dir / "decoder.pt"),
            )

            # Extract bit probabilities — mean prob above 0.5 ≈ message present
            probs = learned_wm.extract(waveform)  # (n_bits,)
            mean_confidence = float(probs.mean().item())

            # Bit accuracy measured against a threshold of 0.5 (random baseline)
            hard_bits = (probs >= 0.5).float()
            # We don't know the original message, so report how far from random
            # (a trained watermark will have consistently high or low probs per bit).
            # detected = mean confidence meaningfully different from 0.5
            deviation = abs(mean_confidence - 0.5)
            detected = deviation > 0.1  # heuristic: 10 % above random baseline

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                watermark_type="learned",
                detected=bool(detected),
                confidence=float(mean_confidence),
                bit_accuracy=float(hard_bits.mean().item()),
                processing_time_ms=elapsed_ms,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                watermark_type="learned",
                detected=False,
                confidence=0.0,
                processing_time_ms=elapsed_ms,
                error=str(exc),
            )

    def detect_all(
        self,
        waveform: torch.Tensor,
        watermark_id: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> List[DetectionResult]:
        """Run all available detectors on *waveform*.

        Args:
            waveform: 1-D float32 audio tensor.
            watermark_id: When provided, also runs classical detection.
                          When *None*, only the learned detector is run.
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of :class:`DetectionResult` — one per detector attempted.
        """
        results: List[DetectionResult] = []

        if watermark_id is not None:
            results.append(
                self.detect_classical(waveform, watermark_id, sample_rate=sample_rate)
            )

        results.append(self.detect_learned(waveform, sample_rate=sample_rate))

        return results

    def detect_from_file(
        self,
        path: str,
        watermark_id: Optional[str] = None,
    ) -> List[DetectionResult]:
        """Read a WAV file and run all detectors.

        Handles corrupted / missing files gracefully by returning a single
        :class:`DetectionResult` with the ``error`` field populated.

        Args:
            path: Path to a 16-bit PCM WAV file.
            watermark_id: Forwarded to :meth:`detect_all`.

        Returns:
            List of :class:`DetectionResult`.
        """
        t0 = time.perf_counter()
        try:
            from ..voice.audio import load_wav, AudioConfig

            # Relaxed config: short clips are fine in tests
            cfg = AudioConfig(min_duration_seconds=0.001, normalize_amplitude=False)
            waveform, sample_rate = load_wav(path, config=cfg)
            return self.detect_all(waveform, watermark_id=watermark_id, sample_rate=sample_rate)

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return [
                DetectionResult(
                    watermark_type="none",
                    detected=False,
                    confidence=0.0,
                    watermark_id=watermark_id,
                    processing_time_ms=elapsed_ms,
                    error=str(exc),
                )
            ]
