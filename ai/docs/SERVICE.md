# EchoSelf Unified AI Service Subsystem

## Overview

The EchoSelf AI Service subsystem exposes a unified, zero-dependency Python service layer that integrates all from-scratch AI models (Language Model, Voice Synthesis, Visual Avatars, and Prompt-to-Video) into a single typed interface.

It is designed to serve as the direct backend inference engine for the Go backend `processor.Processor` interface.

```text
ai/src/echoself_ai/service/
├── __init__.py
├── unified.py     # EchoSelfAIService, ProcessRequest, ProcessResult
├── server.py      # Threaded HTTP server (GET /health, GET /supported_jobs, POST /process)
└── pipeline.py    # MultimodalPipeline (Digital Twin synchronized text -> voice -> video)
```

---

## Architecture & Go Backend Protocol

The service layer adheres to the exact request and result contract expected by the EchoSelf Go backend:

### `ProcessRequest`
```json
{
  "job_id": "c1f7b0a8-...",
  "job_type": "llm_generate | voice_synthesis | avatar_generate | video_generate",
  "user_id": "a9d3e4f1-...",
  "media_id": "optional-uuid",
  "payload": {
    "prompt": "Hello EchoSelf",
    "speaker_id": "optional-speaker",
    "avatar_id": "optional-avatar"
  }
}
```

### `ProcessResult`
```json
{
  "job_id": "c1f7b0a8-...",
  "status": "success | failed",
  "output": { ... },
  "metadata": {
    "execution_time_ms": 142.5,
    "device": "cpu"
  },
  "error": null
}
```

---

## Supported Job Types

| Job Type | Modality | Key Payload Fields | Output Fields |
|---|---|---|---|
| `llm_generate` / `text_generation` | Text / Language Model | `prompt`, `max_new_tokens`, `temperature`, `top_k` | `prompt`, `generated_text`, `num_tokens` |
| `voice_synthesis` | Audio / Voice AI | `text`, `speaker_id` (requires consent) | `audio_base64`, `sample_rate`, `num_samples`, `duration_seconds` |
| `voice_embedding` | Audio Features | `speaker_id` | `speaker_id`, `embedding_dim`, `embedding` |
| `avatar_generate` | Visual AI | `avatar_id` (requires consent), `condition_vector` | `avatar_id`, `image_base64`, `width`, `height` |
| `avatar_embedding` | Visual Features | `avatar_id` | `avatar_id`, `embedding_dim`, `embedding` |
| `video_generate` | Prompt-to-Video | `prompt`, `avatar_id` (optional), `num_frames`, `fps` | `prompt`, `video_gif_base64`, `num_frames`, `fps`, `duration_seconds` |
| `consent_verify` | Governance | `speaker_id`, `avatar_id`, `scope` | `voice_authorized`, `visual_authorized` |
| `consent_register` | Governance | `modality`, `speaker_id`, `avatar_id`, `scope` | `records`, `status` |
| `consent_revoke` | Governance | `modality`, `speaker_id`, `avatar_id` | `revocations`, `status` |
| `consent_purge` | GDPR Erasure | `speaker_id`, `avatar_id` | `purged_records`, `status` |

---

## REST API Endpoints

The HTTP server layer runs via `ThreadedEchoSelfServer` with zero external web dependencies:

- `GET /health`:
  ```json
  {
    "status": "healthy",
    "service": "EchoSelf AI Engine",
    "version": "0.1.0",
    "device": "cpu",
    "modalities": ["text", "voice", "vision", "video"]
  }
  ```
- `GET /supported_jobs`: Returns the list of all active registered job types.
- `POST /process`: Dispatches any `ProcessRequest` to the underlying engine and returns typed `ProcessResult`.

---

## End-to-End Digital Twin Orchestration (`MultimodalPipeline`)

The `MultimodalPipeline` chains together the three human-like output modalities:
1. **Cognitive / Language**: Autoregressively generates conversational responses given incoming context.
2. **Vocal / Audio**: Synthesizes expressive 16-bit PCM WAV speech matching the generated text.
3. **Visual / Kinetic**: Synthesizes temporal animated video frames of the avatar delivering the communication.

**Strict Security & Consent Invariant**: No personalized digital twin operation can execute if either the voice or avatar identity lacks verified authorization from `VoiceConsentManager` or `VisualConsentManager`.

---

## Standalone Server Execution

```bash
# Start server locally
python -m echoself_ai.service.server --host 127.0.0.1 --port 8000 --device cpu

# Using uv
uv run python -m echoself_ai.service.server --host 127.0.0.1 --port 8000
```
