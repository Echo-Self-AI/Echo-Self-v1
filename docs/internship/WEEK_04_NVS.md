# Week 4 — Neural Voice Synthesis (NVS)

## Objective

Generate speech in an enrolled user's voice by reusing the existing acoustic model and Griffin-Lim vocoder.

## Model Selection

The existing `AcousticModel` checkpoint at `ai/checkpoints/acoustic_real/best_model.pt` was reused directly. No new model was trained.

| Component | Implementation | Checkpoint |
|-----------|---------------|------------|
| Tokenizer | `CharacterTokenizer` (existing) | — |
| Acoustic Model | `AcousticModel` (3-layer ConvBlock encoder, speaker embedding) | `checkpoints/acoustic_real/best_model.pt` (1.26 MB) |
| Vocoder | `GriffinLimVocoder` (80-iteration Griffin-Lim) | — (no weights needed) |

## Implementation

### Files Created / Modified

- `ai/src/echoself_ai/voice/nvs.py` — NVS module
- `ai/tests/test_nvs.py` — test suite

### Architecture

**`TextToSpeech`** class:
```
text: str
  → CharacterTokenizer.encode() → token_ids: Tensor[L]
  → AcousticModel(token_ids, speaker_id) → mel: Tensor[80, T]
  → GriffinLimVocoder(mel) → waveform: Tensor[N]
  → normalize_waveform(peak=0.95)
  → (waveform, sample_rate=16000)
```

Synthesis pipeline:
1. Text is character-tokenized (each ASCII character → index)
2. `AcousticModel` predicts 80-channel log-mel spectrogram frames
3. Upsampling factor = 4 (each token generates 4 mel frames)
4. `GriffinLimVocoder` runs 80 iterations of Griffin-Lim algorithm
5. Output is peak-normalized to 0.95 to prevent clipping

**`NVSConfig`** dataclass:
```python
sample_rate: int = 16000
speaker_id: int = 0
normalize: bool = True
target_peak: float = 0.95
```

**`synthesize_to_file(text, output_path)`**: convenience function that calls `synthesize()` and writes WAV file.

**`get_synthesis_latency(text)`**: measures wall-clock synthesis time in seconds.

## Test Results

```
tests/test_nvs.py — passed
```

Coverage:
- `synthesize()` returns `(Tensor, int)` tuple
- Output waveform has correct shape (1D float32)
- No NaN or Inf in output
- `synthesize_to_file()` creates valid WAV file
- Latency measurement returns positive float
- Empty text handling (returns empty waveform gracefully)
- With `speaker_id=0`
- Consent gate integration: raises `PermissionError` when no consent

## Measured Performance (actual)

| Text | Output Duration | Peak Amplitude | NaN | Latency |
|------|----------------|----------------|-----|---------|
| "hello" | 0.152 s | 0.95 | No | 11.47 ms |
| "echo self" | 0.280 s | 0.95 | No | 14.08 ms |
| "testing one two three" | 0.664 s | 0.95 | No | 25.33 ms |
| "voice synthesis" | 0.472 s | 0.95 | No | 18.82 ms |
| "neural audio" | 0.376 s | 0.95 | No | 16.96 ms |

**Mean synthesis latency**: 17.33 ms ± 4.72 ms

## Note on Voice Quality

The acoustic model was trained for a small number of steps on LJSpeech data. The Griffin-Lim vocoder produces intelligible but not natural-sounding speech (spectral smearing is a known Griffin-Lim artifact). Quality would improve significantly with a neural vocoder (e.g., HiFi-GAN) trained on more data.

## Commit

`132d77e` — `feat(nvs): implement neural voice synthesis with acoustic model and Griffin-Lim vocoder`
