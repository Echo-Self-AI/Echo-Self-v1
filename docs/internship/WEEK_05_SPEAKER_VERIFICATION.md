# Week 5 — Voice Enrollment and Speaker Verification

## Objective

Implement voice enrollment (creating a speaker profile from audio samples) and speaker verification (comparing a query voice against an enrolled profile).

## Model Selection

| Option | Architecture | Deps | Decision |
|--------|-------------|------|----------|
| SpeechBrain ECAPA-TDNN | TDNN + attentive pooling | speechbrain package | Not installed |
| d-vector (custom) | Conv1d + BN + temporal avg pool | PyTorch only | ✅ Selected |

A lightweight d-vector extractor was implemented from scratch in PyTorch, inspired by the ECAPA-TDNN concept but significantly simplified for local hardware.

## Implementation

### Files Created

- `ai/src/echoself_ai/voice/speaker_verification.py`
- `ai/tests/test_speaker_verification.py`

### Architecture

**`SpeakerEncoder`** (d-vector extractor):
```
Input: log-mel spectrogram (B, n_mels=80, T)
  → Conv1dBlock × 3 (Conv1d + BatchNorm1d + ReLU, kernel=5, same-padding)
    each: input_dim → hidden_dim=128
  → Temporal average pooling: (B, 128, T) → (B, 128)
  → Linear(128, 192) → ReLU
  → Linear(192, embedding_dim=192)
  → F.normalize(p=2, dim=-1)
Output: (B, 192) L2-normalized d-vector
```

**`SpeakerProfile`** dataclass:
```python
user_id: str
enrollment_id: str          # UUID
mean_embedding: list[float] # embedding_dim floats
enrolled_at: str            # ISO timestamp
num_samples: int
```

**`SpeakerEnrollment`**:
- `enroll(user_id, waveform_list, sample_rate)` → `SpeakerProfile`
  - Extracts embedding for each waveform via `LogMelSpectrogram` + `SpeakerEncoder`
  - Averages embeddings and L2-normalizes the mean
  - Persists as JSON to `ai/data/speakers/<user_id>.json`
  - Sanitizes user_id to prevent path traversal
- `revoke(user_id)` → deletes profile file
- `load_profile(user_id)` → `Optional[SpeakerProfile]`

**`SpeakerVerifier`**:
- `verify(user_id, waveform, sample_rate, threshold=0.75)` → `(bool, float)`
  - Loads enrolled profile
  - Extracts embedding from query waveform
  - Computes cosine similarity (dot product of L2-normalized vectors)
  - Returns `(similarity >= threshold, similarity)`
  - Returns `(False, 0.0)` if no profile exists

**`generate_challenge_phrase()`**: returns random phrase from list of 20 challenge phrases.

## Security

- Speaker profiles stored in `ai/data/speakers/` (never in a public directory)
- Embeddings stored as base64-encoded numpy bytes — not human-readable
- User IDs sanitized: only alphanumeric + hyphen/underscore allowed
- Profiles never exposed through public endpoints
- Revocation immediately deletes the profile file

## Test Results

```
tests/test_speaker_verification.py — 45 passed
```

9 test classes covering:
- `SpeakerEncoderConfig` validation
- `SpeakerEncoder` forward pass: shape `(B, 192)`, L2 norm ≈ 1.0
- `SpeakerProfile` serialization round-trip
- `SpeakerEnrollment`: file created, multi-sample, revoke
- Same-input similarity > 0.5
- `SpeakerVerifier` accepts/rejects at thresholds
- Challenge phrase is non-empty string from known list

## Measured Performance (actual)

| Scenario | Similarity |
|----------|-----------|
| Same waveform enrolled and queried | 1.000 |
| Same frequency, different amplitude | 0.998 |
| Different frequency (4000 Hz vs 440 Hz) | 0.914 |
| Matched (3 enrollment samples, same wave) | 1.000 |
| Unmatched (different generation) | 0.899 |

**Note on FAR/FRR**: The d-vector encoder is **untrained** (random weights), so similarity values reflect random embedding geometry rather than true voice identity. After training on speaker data (e.g., VoxCeleb), FAR/FRR would be meaningful metrics. At default threshold 0.75, the system accepts any input with untrained weights since similarities cluster around 0.9 for synthetic data.

## Commit

`8d6667c` — `feat(speaker-verification): implement speaker enrollment and verification with d-vector embeddings`
