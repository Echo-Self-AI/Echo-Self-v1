# EchoSelf — Known Limitations

## Hardware Constraints

| Constraint | Impact |
|-----------|--------|
| 3.7 GB VRAM (RTX 3050) | Limits model size; large neural vocoders (HiFi-GAN full) cannot be trained locally |
| 7.8 GB system RAM | Learned watermark trained on tiny synthetic dataset only |
| No persistent CUDA memory across training runs | Each training session starts fresh |

## ASR Quality

The CTC ASR model (`asr.py`) implements a correct architecture but has **no trained weights**. Without training on a labeled speech-text dataset (LibriSpeech, Common Voice), it produces random characters rather than accurate transcripts.

**Workaround**: The `SpeechRecognizer` interface is designed to be a drop-in replacement. Swapping in Whisper (via `whisper` package) or Vosk requires only replacing the `recognize()` method body.

## Speaker Verification Accuracy

The `SpeakerEncoder` has **random initial weights** (not trained on speaker data). Key consequences:
- Similarity scores do not reflect voice identity
- FAR and FRR cannot be measured meaningfully
- EER is undefined without a labeled multi-speaker evaluation set

Proper training would require VoxCeleb or a similar speaker corpus and multiple training epochs.

## Classical Watermark Robustness

Classical spread-spectrum watermarking **breaks under temporal modifications**:
- Trimming (removing samples from start/end)
- Padding (adding samples)
- Time-stretching
- Different-length re-encoding

This is a fundamental property of time-domain PN-sequence correlation and is not fixable without switching to a transform-domain approach (DCT, DWT).

## Learned Watermark Scale

The learned watermark was trained for 200 steps on synthetic random audio. This is a proof-of-concept:
- Bit accuracy slightly above 50% (not production-quality)
- Not tested on real speech
- Full-scale training (10k+ steps, real speech, data augmentation) required for practical use

## Voice Quality

Griffin-Lim vocoder produces audible but not natural-sounding speech:
- Spectral smearing artifact (characteristic of Griffin-Lim)
- Requires a neural vocoder (WaveGlow, HiFi-GAN) for production quality
- Acoustic model was trained for limited epochs on LJSpeech

## Audit Log Security

The append-only audit log uses **structural JSON validation** for tamper detection, not cryptographic signatures. A sophisticated attacker who can modify the file can:
- Delete entire records (not detected)
- Replace valid JSON lines with different valid JSON (not detected)

A production system should use HMAC or a Merkle chain for strong tamper evidence.

## Real-time Streaming

The pipeline is **batch-mode**, not streaming. It processes a complete audio file and returns a complete output. True real-time streaming would require:
- Chunk-by-chunk VAD with ring buffer
- Incremental ASR (supported by some models)
- Streaming vocoder
- WebSocket or SSE transport

## Go Backend Connectivity

The Go backend integration requires PostgreSQL and Redis (via Docker). The integration tests in `backend/internal/processor/` test the HTTP bridge but require the Python service to be running. Full end-to-end Go→Python tests are only possible in a Docker environment.
