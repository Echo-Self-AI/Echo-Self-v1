# Week 8 — Watermark Detector and Audit Logging

## Objective

Build an independent watermark detector that can process audio files and report detection results, and an append-only tamper-evident audit logging system.

## Watermark Detector

### Files Created

- `ai/src/echoself_ai/watermark/detector.py`
- `ai/tests/test_detector.py`

### `DetectionResult` dataclass

```python
watermark_type: str          # 'classical', 'learned', 'none', 'unsupported'
detected: bool
confidence: float            # 0.0–1.0
watermark_id: Optional[str]
correlation: Optional[float]
bit_accuracy: Optional[float]
processing_time_ms: float
error: Optional[str]         # set if input was corrupt/invalid
```

### `WatermarkDetector` class

- `detect_classical(waveform, watermark_id, sample_rate)` → `DetectionResult`
  - Calls `SpreadSpectrumWatermarker.extract()`
  - Maps normalized correlation to confidence score
- `detect_learned(waveform, sample_rate)` → `DetectionResult`
  - Uses `LearnedWatermarker` if checkpoint exists, else returns `unsupported`
- `detect_all(waveform, watermark_id, sample_rate)` → `list[DetectionResult]`
  - Runs all available detectors
- `detect_from_file(path, watermark_id)` → `list[DetectionResult]`
  - Reads WAV via `audio.py::ingest_audio()`
  - Wraps in try/except — corrupt audio → `DetectionResult` with error field

## Audit Logger

### Files Created

- `ai/src/echoself_ai/audit/__init__.py`
- `ai/src/echoself_ai/audit/logger.py`
- `ai/tests/test_audit.py`

### `AuditRecord` dataclass

```python
event_id: str           # UUID4
user_id: Optional[str]
timestamp: str          # ISO 8601
model_id: str
watermark_type: str
watermark_id: Optional[str]
output_hash: Optional[str]  # SHA-256 hex of PCM bytes
detection_result: Optional[bool]
verification_status: Optional[bool]
action: str             # 'synthesize' | 'detect' | 'verify' | 'enroll' | 'revoke'
metadata: dict
```

### `AuditLogger` class

- Log file: `ai/data/audit/audit_log.jsonl` (JSON Lines, one record per line)
- Directory created with `mode=0o700` (owner-only access)
- `log(record)` → appends JSON line (append-only, never overwrites)
- `read_all()` → parses all lines, returns `list[AuditRecord]`
- `verify_integrity()` → reads all lines, returns `True` if all valid JSON
- `export_csv(path)` → writes CSV of all records

**`compute_audio_hash(waveform, sample_rate)`**: SHA-256 of raw 16-bit PCM bytes, returns 64-char hex string.

## Test Results

```
tests/test_detector.py — passed
tests/test_audit.py — passed
```

Audit log tests:
- 5 records logged → `read_all()` returns 5
- `verify_integrity()` returns `True` on clean log
- After manually corrupting a line → `verify_integrity()` returns `False`
- `compute_audio_hash()` returns 64-char hex
- Same waveform → same hash; different waveforms → different hashes
- `export_csv()` creates valid CSV

## Security Properties

| Property | Implementation |
|----------|---------------|
| Append-only | File opened with `'a'` mode; no delete/overwrite methods |
| Tamper detection | `verify_integrity()` checks all lines are valid JSON |
| No raw audio | Only SHA-256 hash of PCM bytes stored |
| Access restriction | Log directory created with `mode=0o700` |
| Sensitive data | user_id referenced but personal audio never stored |

## Commit

`879b266` — `feat(detector,audit): implement watermark detector and append-only audit logging system`
