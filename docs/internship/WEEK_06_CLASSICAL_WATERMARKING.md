# Week 6 — Classical Audio Watermarking

## Objective

Implement a spread-spectrum baseline watermarking system for synthesized audio.

## Algorithm

**Additive spread-spectrum watermarking**:
```
embed:    watermarked = waveform + α × pn_sequence
extract:  correlation = dot(waveform, pn_sequence) / (||waveform|| × ||pn_sequence||)
detect:   detected = correlation > threshold (0.001)
```

Where `pn_sequence` is a pseudorandom {-1, +1} sequence derived deterministically from `watermark_id` via SHA-256 → `torch.Generator` seeding. This ensures the PN sequence is reproducible without global random state pollution.

## Implementation

### Files Created

- `ai/src/echoself_ai/watermark/classical.py`
- `ai/src/echoself_ai/watermark/config.py`
- `ai/src/echoself_ai/watermark/__init__.py`
- `ai/tests/test_classical_watermark.py`
- `ai/scripts/benchmark_watermark.py`

### Key Classes

**`ClassicalWatermarkConfig`**:
```python
watermark_id: str = ""    # UUID string
strength: float = 0.01    # alpha (embedding amplitude)
n_bits: int = 32
seed: int = 42
sample_rate: int = 16000
```

**`SpreadSpectrumWatermarker`**:
- `embed(waveform, watermark_id)` → watermarked tensor (same shape, clipped to [-1,1])
- `extract(waveform, watermark_id)` → `(detected: bool, correlation: float, ber: float)`
- `compute_snr(original, watermarked)` → SNR in dB

**`n_bit_watermark_id_to_bits(watermark_id, n_bits)`**: SHA-256 hash → n binary bits (deterministic)
**`bit_error_rate(original_bits, extracted_bits)`**: fraction of differing bits

## Test Results

```
tests/test_classical_watermark.py — 22 passed
```

## Measured Performance (actual)

| Strength (α) | SNR (dB) | Detection Rate | Embed Latency | Detect Latency |
|-------------|---------|----------------|---------------|----------------|
| 0.001 | 55.05 | 0% (below threshold) | 0.48 ms | 0.49 ms |
| 0.01 | 35.05 | 100% | 0.35 ms | 0.47 ms |
| 0.05 | 21.07 | 100% | 0.31 ms | 0.47 ms |

At `strength=0.01` (default): SNR = 35 dB, fully imperceptible, 100% detection on clean audio.

**After Gaussian noise (σ=0.01)**: 100% detection maintained at strength=0.01 and 0.05.

**Robustness to trimming/padding**: detection fails after trimming (10% removed) or padding, since the PN sequence alignment is broken. This is a known limitation of time-domain spread-spectrum.

## Commit

`87bfd2a` — `feat(watermark): implement spread-spectrum classical audio watermarking`
