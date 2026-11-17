# Week 2 — Voice Activity Detection (VAD)

## Objective

Develop a modular Voice Activity Detection module that identifies speech segments in audio, with configurable thresholds, hangover, and minimum speech duration.

## Research Summary

Three approaches were evaluated:

| Approach | Pros | Cons |
|----------|------|------|
| Energy-based VAD | Zero extra dependencies, deterministic, <1 ms latency | Sensitive to loud noise |
| WebRTC VAD | Well-tuned for telephony | Requires `webrtcvad` C extension (not installed) |
| Silero VAD | High accuracy, noise-robust | Requires model download, ~1.8 MB |

**Decision**: Energy-based VAD was selected as it requires no additional dependencies, runs entirely in PyTorch, and is sufficient for clean-to-moderate-noise speech detection. The architecture is modular so Silero can be swapped in later.

## Implementation

### Files Created

- `ai/src/echoself_ai/voice/vad.py` — main VAD implementation
- `ai/src/echoself_ai/voice/vad_config.py` — `VADConfig` dataclass
- `ai/tests/test_vad.py` — test suite
- `ai/scripts/benchmark_vad.py` — benchmark script

### Architecture

**`EnergyVAD`** (pure PyTorch):
1. Splits waveform into fixed-length frames (10/20/30 ms configurable)
2. Computes RMS energy per frame
3. Applies energy threshold (configurable, default 0.01)
4. Applies hangover (frames after last speech, default 300 ms)
5. Merges contiguous speech frames into segments
6. Filters segments shorter than `min_speech_duration_ms` (default 100 ms)

**`VoiceActivityDetector`** (unified interface):
- `detect(waveform, sample_rate)` → `list[tuple[float, float]]` (start/end seconds)
- `segment_waveform(waveform, sample_rate)` → `list[Tensor]`
- `is_speech_frame(frame)` → `bool`
- `create_vad(method='energy')` factory function

**`VADConfig`** dataclass:
```python
frame_duration_ms: int = 20       # 20 ms frames
energy_threshold: float = 0.01    # RMS threshold
min_speech_duration_ms: float = 100
hangover_ms: float = 300
sample_rate: int = 16000
```

## Test Results

```
tests/test_vad.py — 42 passed
```

Coverage:
- Silence-only audio → 0 segments (no false positives)
- Tone-only audio → 1 segment detected
- Tone-silence-tone pattern → 2 segments
- Very short audio (100 ms) → handled without crash
- Empty/zero audio → 0 segments
- NaN audio → raises or returns empty
- Min speech duration filter
- Multiple frame size configurations (10, 20, 30 ms)

## Benchmark Results (actual measurements)

| Audio Type | Segments Found | Latency | Real-time Factor |
|------------|----------------|---------|------------------|
| 2s silence | 0 | 0.49 ms | 0.000245 |
| 2s pure tone | 1 | 0.15 ms | 0.000075 |
| tone+gap+tone | 2 | 0.20 ms | 0.000100 |

VAD runs >1000× faster than real-time on all tested audio types.

## Commit

`88750cb` — `feat(vad): implement energy-based voice activity detection module`
