# EchoSelf — Abstract

**EchoSelf: Your Voice and Identity, Echoed Digitally**

EchoSelf is a local-first, privacy-preserving AI system for voice identity capture, speech synthesis, and provenance tracking. It enables users to create a faithful digital twin of their voice, authenticate speech content, and audit all synthesis events — entirely on-device without transmitting audio to external services.

The core pipeline processes audio through seven integrated stages: Voice Activity Detection (VAD) segments speech from silence; a CTC-based Automatic Speech Recognition (ASR) module transcribes speech to text; a Neural Voice Synthesis (NVS) module generates speech via an acoustic model and Griffin-Lim vocoder; a spread-spectrum and learned watermarking system embeds imperceptible provenance marks into synthesized audio; an independent watermark detector verifies embeds; and an append-only tamper-evident audit logger records every synthesis event.

Speaker verification is performed using lightweight d-vector embeddings extracted by a convolutional speaker encoder. Enrollment stores speaker profiles locally with base64-encoded embeddings in JSON; verification uses cosine similarity against enrolled means.

Classical spread-spectrum watermarking achieves SNR > 35 dB at strength 0.01, with 100% detection rate on clean audio. Learned watermarking — a compact Conv1d encoder-decoder trained entirely within the system — achieves >50% bit accuracy on clean audio as a proof-of-concept appropriate for the available hardware (RTX 3050, 3.7 GB VRAM).

The full end-to-end pipeline runs at 0.008× real-time (>120× faster than real-time), with NVS dominating latency at ~14 ms per utterance. The system passes 631 automated Python tests and enforces consent gates at every voice-access boundary.

This internship project was implemented entirely in Python and PyTorch using only free, open-source tools with ₹0 paid API or cloud expenditure, running locally on consumer hardware.
