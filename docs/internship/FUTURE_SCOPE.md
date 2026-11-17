# EchoSelf — Future Scope

## Speech Recognition

**Short term**: Integrate Whisper Tiny (39 MB, MIT license) for real ASR. The `SpeechRecognizer` interface already matches the expected Whisper API contract.

**Medium term**: Fine-tune Whisper on user's own speech for personalized ASR with improved accuracy on the enrolled speaker's vocabulary and speaking style.

## Neural Voice Synthesis

**Short term**: Replace Griffin-Lim with a lightweight neural vocoder. HiFi-GAN (generator only, ~14 MB) is MIT-licensed and runs on CPU in ~20 ms per second of audio.

**Medium term**: Implement a proper multi-speaker TTS system using speaker embeddings from `SpeakerEncoder` to condition synthesis — enabling true personalized voice cloning after enrollment.

**Long term**: Train on the enrolled user's own recordings (with consent) to produce natural-sounding output in their voice.

## Speaker Verification

**Short term**: Train `SpeakerEncoder` on Mozilla Common Voice or a subset of VoxCeleb to produce meaningful FAR/FRR measurements.

**Medium term**: Implement speaker diarization (who spoke when) for multi-speaker scenarios.

**Long term**: Anti-spoofing countermeasures (detect synthesized/replayed audio as enrollment material).

## Watermarking

**Short term**: Train the learned watermark for 10k+ steps on real speech data to achieve >90% bit accuracy.

**Medium term**: Implement transform-domain watermarking (DCT/DWT-based) to achieve robustness against trimming and padding.

**Long term**: Investigate neural audio codec watermarking (similar to AudioSeal or WavMark) for perceptually lossless marking robust to common transformations including MP3 compression.

## Audit and Compliance

**Short term**: Add HMAC-based integrity protection to audit records (each line signed with a rotating key).

**Medium term**: Build a compliance dashboard showing all synthesis events, watermark detections, and verification outcomes per user.

**Long term**: Integrate with a transparency log (Certificate Transparency-style) for public verifiable audit trails.

## Pipeline

**Short term**: Implement true streaming mode using WebSocket transport and chunk-by-chunk VAD.

**Medium term**: Add concurrency control (bounded worker pool) so multiple users can synthesize simultaneously without VRAM contention.

**Long term**: Edge deployment — package the pipeline as a local binary (using PyInstaller or a similar tool) with a minimal web UI.

## Security

**Short term**: Add replay attack detection to speaker verification (require a fresh challenge phrase to be spoken as part of enrollment/verification).

**Medium term**: Encrypt speaker profiles at rest using the OS keyring (secretservice on Linux, Keychain on macOS).

**Long term**: Differential privacy guarantees on speaker embeddings — prevent embedding inversion attacks.

## Evaluation

**Short term**: Benchmark on LibriSpeech test-clean to measure ASR Word Error Rate once a real ASR model is integrated.

**Medium term**: Measure MOS (Mean Opinion Score) on synthesized audio using automated evaluation (UTMOS or similar).

**Long term**: Conduct a user study comparing EchoSelf voice quality against commercial TTS systems.
