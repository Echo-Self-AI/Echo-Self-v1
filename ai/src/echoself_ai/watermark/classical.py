"""
Classical spread-spectrum audio watermarking.

Algorithm overview
------------------
Embedding (additive spread-spectrum):
    watermarked = waveform + alpha * pn_sequence
    where pn_sequence is a pseudorandom {-1, +1} sequence derived
    deterministically from the watermark_id.

Detection (normalised cross-correlation):
    correlation = dot(waveform, pn_sequence) / (||waveform|| * ||pn_sequence||)
    If correlation > detection_threshold  →  watermark detected.

No external libraries (scipy, librosa, …) are used.  Only PyTorch and the
Python standard library.
"""

from __future__ import annotations

import hashlib
import math
from typing import Tuple

import torch

from .config import ClassicalWatermarkConfig

# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------

def n_bit_watermark_id_to_bits(watermark_id: str, n_bits: int) -> list[int]:
    """Hash *watermark_id* deterministically to a list of *n_bits* bits.

    Uses SHA-256 so the mapping is stable across Python processes.

    Args:
        watermark_id: Arbitrary string identifier.
        n_bits: Desired number of bits (must be > 0).

    Returns:
        List of *n_bits* integers, each 0 or 1.
    """
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")

    digest = hashlib.sha256(watermark_id.encode("utf-8")).digest()
    bits: list[int] = []
    for byte in digest:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
            if len(bits) == n_bits:
                return bits

    # SHA-256 produces 256 bits; if n_bits > 256 we cycle the hash output.
    while len(bits) < n_bits:
        extra_digest = hashlib.sha256(
            (watermark_id + str(len(bits))).encode("utf-8")
        ).digest()
        for byte in extra_digest:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
                if len(bits) == n_bits:
                    return bits

    return bits[:n_bits]


def bit_error_rate(original_bits: list[int], extracted_bits: list[int]) -> float:
    """Fraction of bit positions that differ between two bit sequences.

    Args:
        original_bits: Reference bit sequence.
        extracted_bits: Recovered bit sequence (same length expected).

    Returns:
        BER in [0.0, 1.0].  Returns 1.0 when lengths differ.
    """
    if len(original_bits) != len(extracted_bits):
        return 1.0
    if not original_bits:
        return 0.0
    errors = sum(a != b for a, b in zip(original_bits, extracted_bits))
    return errors / len(original_bits)


# ---------------------------------------------------------------------------
# Main watermarker
# ---------------------------------------------------------------------------

class SpreadSpectrumWatermarker:
    """Classical spread-spectrum audio watermarker.

    Args:
        config: :class:`ClassicalWatermarkConfig` instance.
    """

    # Minimum normalised correlation required to claim detection.
    DETECTION_THRESHOLD: float = 0.001

    def __init__(self, config: ClassicalWatermarkConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_pn_sequence(length: int, seed: int) -> torch.Tensor:
        """Generate a pseudorandom PN sequence of *length* {-1, +1} values.

        Uses :func:`torch.manual_seed` so the output is reproducible given the
        same *seed* and *length*.

        Args:
            length: Number of samples in the sequence.
            seed: Integer seed that uniquely determines the sequence.

        Returns:
            1-D float32 tensor of shape ``(length,)`` with values in {-1, +1}.
        """
        generator = torch.Generator()
        generator.manual_seed(seed % (2**63))  # Generator seeds must fit in int64
        # Sample uniform [0, 1) then threshold at 0.5 to get {0, 1}, map to {-1, +1}
        uniform = torch.rand(length, generator=generator, dtype=torch.float32)
        pn = torch.where(uniform < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
        return pn

    @staticmethod
    def _watermark_id_to_seed(watermark_id: str) -> int:
        """Convert a watermark_id string to a reproducible integer seed.

        Uses SHA-256 to ensure good distribution and cross-platform stability.
        """
        digest = hashlib.sha256(watermark_id.encode("utf-8")).digest()
        # Take the first 8 bytes as a big-endian unsigned 64-bit integer.
        seed = int.from_bytes(digest[:8], byteorder="big")
        return seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, waveform: torch.Tensor, watermark_id: str) -> torch.Tensor:
        """Embed a watermark into *waveform* using additive spread-spectrum.

        The watermark is imperceptible at typical strength values (≤ 0.01) but
        detectable via correlation.

        Args:
            waveform: Input audio tensor of arbitrary shape.  The PN sequence
                      is generated to match the total number of elements (i.e.,
                      the flat view).  Shape is preserved in the output.
            watermark_id: String identifier for the watermark.  Must be the
                          same string used during extraction.

        Returns:
            Watermarked waveform tensor with the same shape and dtype as
            *waveform*, clipped to [-1, 1].
        """
        seed = self._watermark_id_to_seed(watermark_id)
        total_samples = waveform.numel()
        pn = self._generate_pn_sequence(total_samples, seed)
        pn = pn.reshape(waveform.shape).to(waveform.dtype)

        watermarked = waveform + self.config.strength * pn
        watermarked = torch.clamp(watermarked, -1.0, 1.0)
        return watermarked

    def extract(
        self,
        waveform: torch.Tensor,
        watermark_id: str,
    ) -> Tuple[bool, float, float]:
        """Attempt to extract a watermark from *waveform*.

        Re-generates the same PN sequence and computes the normalised
        cross-correlation between the waveform and the sequence.

        Args:
            waveform: Audio tensor to test (arbitrary shape).
            watermark_id: Identifier of the watermark to look for.

        Returns:
            A tuple ``(detected, correlation, bit_error_rate)`` where

            * *detected* – ``True`` when
              ``correlation > DETECTION_THRESHOLD``.
            * *correlation* – Normalised cross-correlation in [-1, 1].
            * *bit_error_rate* – BER computed by extracting per-bit decisions
              from the correlation and comparing against the ideal bit pattern.
        """
        seed = self._watermark_id_to_seed(watermark_id)
        total_samples = waveform.numel()
        pn = self._generate_pn_sequence(total_samples, seed)
        pn = pn.reshape(waveform.shape).to(waveform.dtype)

        flat_waveform = waveform.flatten().float()
        flat_pn = pn.flatten().float()

        # Normalised cross-correlation: dot / (||x|| * ||pn||)
        dot = torch.dot(flat_waveform, flat_pn).item()
        norm_w = torch.linalg.norm(flat_waveform).item()
        norm_p = torch.linalg.norm(flat_pn).item()

        if norm_w < 1e-12 or norm_p < 1e-12:
            correlation = 0.0
        else:
            correlation = dot / (norm_w * norm_p)

        detected = correlation > self.DETECTION_THRESHOLD

        # Bit-error-rate: compare expected bits against extracted bit decisions
        # extracted bit = sign of local segment correlation for each bit slot.
        original_bits = n_bit_watermark_id_to_bits(watermark_id, self.config.n_bits)
        extracted_bits = self._extract_bits(flat_waveform, flat_pn, self.config.n_bits)
        ber = bit_error_rate(original_bits, extracted_bits)

        return detected, float(correlation), float(ber)

    def _extract_bits(
        self,
        waveform: torch.Tensor,
        pn: torch.Tensor,
        n_bits: int,
    ) -> list[int]:
        """Extract bit decisions from a watermarked waveform.

        Divides the waveform into *n_bits* segments and computes the sign of
        the local correlation in each segment.  A positive correlation → bit 1;
        negative → bit 0.

        Args:
            waveform: Flat float32 waveform.
            pn: Flat float32 PN sequence (same length as waveform).
            n_bits: Number of bits to extract.

        Returns:
            List of *n_bits* integers, each 0 or 1.
        """
        length = waveform.numel()
        segment_len = max(1, length // n_bits)
        bits: list[int] = []
        for i in range(n_bits):
            start = i * segment_len
            end = min(start + segment_len, length)
            if start >= length:
                bits.append(0)
                continue
            seg_w = waveform[start:end]
            seg_p = pn[start:end]
            local_corr = torch.dot(seg_w, seg_p).item()
            bits.append(1 if local_corr >= 0.0 else 0)
        return bits

    def compute_snr(
        self, original: torch.Tensor, watermarked: torch.Tensor
    ) -> float:
        """Compute Signal-to-Noise Ratio (SNR) in dB.

        SNR = 10 * log10(signal_power / noise_power)

        where *noise_power* is the power of the watermark signal
        (``watermarked - original``) and *signal_power* is the power of the
        original signal.

        Args:
            original: Clean audio tensor.
            watermarked: Watermarked audio tensor (same shape as *original*).

        Returns:
            SNR in dB.  Returns ``inf`` when the noise power is zero (no
            watermark embedded).  Returns ``-inf`` when signal power is zero.
        """
        original_f = original.float().flatten()
        watermarked_f = watermarked.float().flatten()

        noise = watermarked_f - original_f
        signal_power = torch.mean(original_f ** 2).item()
        noise_power = torch.mean(noise ** 2).item()

        if noise_power < 1e-20:
            return math.inf
        if signal_power < 1e-20:
            return -math.inf

        return 10.0 * math.log10(signal_power / noise_power)
