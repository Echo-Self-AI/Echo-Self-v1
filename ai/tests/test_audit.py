"""
Tests for AuditLogger, AuditRecord, and compute_audio_hash.

Coverage:
- log() appends a record
- read_all() returns list with the record
- verify_integrity() returns True for clean log
- After manually corrupting log, verify_integrity() returns False
- compute_audio_hash() returns a 64-char hex string
- Two different waveforms produce different hashes
- Same waveform produces the same hash
- export_csv() creates a valid CSV file
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from echoself_ai.audit.logger import AuditLogger, AuditRecord, compute_audio_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**kwargs) -> AuditRecord:
    """Create a minimal AuditRecord with optional overrides."""
    defaults = dict(
        model_id="test-model-v1",
        watermark_type="classical",
        action="synthesize",
    )
    defaults.update(kwargs)
    return AuditRecord(**defaults)


def _make_logger(tmp_dir: str) -> AuditLogger:
    """Return an AuditLogger backed by a temp directory."""
    log_path = str(Path(tmp_dir) / "audit" / "audit_log.jsonl")
    return AuditLogger(log_path=log_path)


# ---------------------------------------------------------------------------
# compute_audio_hash
# ---------------------------------------------------------------------------

class TestComputeAudioHash:
    """Tests for compute_audio_hash()."""

    def test_returns_64_char_hex_string(self) -> None:
        waveform = torch.rand(16000)
        h = compute_audio_hash(waveform, 16000)
        assert isinstance(h, str)
        assert len(h) == 64
        # Must be valid lowercase hex
        int(h, 16)

    def test_different_waveforms_different_hashes(self) -> None:
        torch.manual_seed(0)
        w1 = torch.rand(16000)
        torch.manual_seed(1)
        w2 = torch.rand(16000)
        h1 = compute_audio_hash(w1, 16000)
        h2 = compute_audio_hash(w2, 16000)
        assert h1 != h2

    def test_same_waveform_same_hash(self) -> None:
        torch.manual_seed(42)
        waveform = torch.rand(16000) * 2.0 - 1.0
        h1 = compute_audio_hash(waveform, 16000)
        h2 = compute_audio_hash(waveform, 16000)
        assert h1 == h2

    def test_sample_rate_does_not_affect_hash(self) -> None:
        """Hash is derived from PCM bytes only, not from sample rate."""
        torch.manual_seed(7)
        waveform = torch.rand(8000)
        h1 = compute_audio_hash(waveform, 16000)
        h2 = compute_audio_hash(waveform, 8000)
        assert h1 == h2

    def test_multidimensional_tensor(self) -> None:
        """Multi-dim tensors should be flattened before hashing."""
        torch.manual_seed(5)
        w1d = torch.rand(1000)
        w2d = w1d.unsqueeze(0)  # (1, 1000)
        assert compute_audio_hash(w1d, 16000) == compute_audio_hash(w2d, 16000)

    def test_hash_length_independent_of_waveform_length(self) -> None:
        """SHA-256 always produces 64 hex chars."""
        for n in [100, 1000, 16000]:
            w = torch.rand(n)
            assert len(compute_audio_hash(w, 16000)) == 64


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------

class TestAuditRecord:
    """Tests for AuditRecord dataclass construction and serialisation."""

    def test_default_uuid_event_id(self) -> None:
        rec = _make_record()
        # Should be a UUID string (36 chars with dashes)
        assert len(rec.event_id) == 36
        assert rec.event_id.count("-") == 4

    def test_default_timestamp_is_iso(self) -> None:
        rec = _make_record()
        # Should parse without error
        dt = datetime.fromisoformat(rec.timestamp)
        assert dt.tzinfo is not None

    def test_to_dict_roundtrip(self) -> None:
        rec = _make_record(user_id="user-42", watermark_id="wm-1")
        d = rec.to_dict()
        rec2 = AuditRecord.from_dict(d)
        assert rec.event_id == rec2.event_id
        assert rec.user_id == rec2.user_id
        assert rec.watermark_id == rec2.watermark_id

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match="action"):
            _make_record(action="invalid_action")

    def test_valid_actions(self) -> None:
        for action in ("synthesize", "detect", "verify", "enroll", "revoke"):
            rec = _make_record(action=action)
            assert rec.action == action

    def test_metadata_defaults_to_empty_dict(self) -> None:
        rec = _make_record()
        assert isinstance(rec.metadata, dict)
        assert len(rec.metadata) == 0


# ---------------------------------------------------------------------------
# AuditLogger.log / read_all
# ---------------------------------------------------------------------------

class TestAuditLoggerLog:
    """Tests for log() and read_all()."""

    def test_log_appends_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            rec = _make_record(action="detect")
            logger.log(rec)

            # Read the raw file and check a line was added
            lines = logger._log_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1

    def test_read_all_returns_logged_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            rec = _make_record(user_id="alice", action="synthesize")
            logger.log(rec)

            all_records = logger.read_all()
            assert len(all_records) == 1
            assert all_records[0].user_id == "alice"
            assert all_records[0].action == "synthesize"
            assert all_records[0].event_id == rec.event_id

    def test_multiple_logs_accumulate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            for i in range(5):
                logger.log(_make_record(model_id=f"model-{i}"))

            records = logger.read_all()
            assert len(records) == 5
            model_ids = [r.model_id for r in records]
            assert model_ids == [f"model-{i}" for i in range(5)]

    def test_log_creates_file_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            assert not logger._log_path.exists()
            logger.log(_make_record())
            assert logger._log_path.exists()

    def test_log_creates_directory_with_restricted_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            logger.log(_make_record())
            log_dir = logger._log_path.parent
            mode = oct(os.stat(log_dir).st_mode)
            # On Linux the directory should have 0o700 permissions
            assert "700" in mode, f"Expected 700 permissions, got {mode}"

    def test_read_all_empty_log_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            assert logger.read_all() == []

    def test_each_line_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            for _ in range(3):
                logger.log(_make_record())

            lines = logger._log_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                json.loads(line)  # Should not raise


# ---------------------------------------------------------------------------
# AuditLogger.verify_integrity
# ---------------------------------------------------------------------------

class TestAuditLoggerVerifyIntegrity:
    """Tests for verify_integrity()."""

    def test_clean_log_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            for _ in range(3):
                logger.log(_make_record())

            assert logger.verify_integrity() is True

    def test_missing_log_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            assert logger.verify_integrity() is True

    def test_corrupted_log_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            logger.log(_make_record())
            logger.log(_make_record())

            # Manually corrupt the second line by appending junk
            content = logger._log_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            lines[1] = lines[1] + "CORRUPTED_GARBAGE{{{not json"
            logger._log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            assert logger.verify_integrity() is False

    def test_truncated_json_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            logger.log(_make_record())

            # Truncate the JSON object mid-way
            raw = logger._log_path.read_bytes()
            logger._log_path.write_bytes(raw[:len(raw) // 2])

            assert logger.verify_integrity() is False

    def test_empty_lines_are_ignored(self) -> None:
        """Blank lines in the log file should not cause verify_integrity to fail."""
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            logger.log(_make_record())
            # Insert a blank line
            content = logger._log_path.read_text(encoding="utf-8")
            logger._log_path.write_text(content + "\n\n", encoding="utf-8")
            assert logger.verify_integrity() is True


# ---------------------------------------------------------------------------
# AuditLogger.export_csv
# ---------------------------------------------------------------------------

class TestAuditLoggerExportCsv:
    """Tests for export_csv()."""

    def test_export_creates_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            for i in range(3):
                logger.log(_make_record(model_id=f"model-{i}"))

            csv_path = str(Path(tmp) / "export.csv")
            logger.export_csv(csv_path)
            assert Path(csv_path).is_file()

    def test_exported_csv_has_correct_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            rec = _make_record(user_id="bob", action="detect", model_id="m1")
            logger.log(rec)

            csv_path = str(Path(tmp) / "export.csv")
            logger.export_csv(csv_path)

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert rows[0]["user_id"] == "bob"
            assert rows[0]["action"] == "detect"
            assert rows[0]["model_id"] == "m1"

    def test_exported_csv_header_has_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            logger.log(_make_record())

            csv_path = str(Path(tmp) / "export.csv")
            logger.export_csv(csv_path)

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames

            expected = {
                "event_id", "user_id", "timestamp", "model_id",
                "watermark_type", "watermark_id", "output_hash",
                "detection_result", "verification_status", "action", "metadata",
            }
            assert expected.issubset(set(fieldnames))

    def test_export_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            csv_path = str(Path(tmp) / "empty.csv")
            logger.export_csv(csv_path)
            assert Path(csv_path).is_file()

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows == []

    def test_metadata_serialised_as_json_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            rec = _make_record(metadata={"key": "value", "count": 3})
            logger.log(rec)

            csv_path = str(Path(tmp) / "export.csv")
            logger.export_csv(csv_path)

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            meta = json.loads(rows[0]["metadata"])
            assert meta["key"] == "value"
            assert meta["count"] == 3

    def test_multiple_records_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)
            for i in range(10):
                logger.log(_make_record(model_id=f"m{i}"))

            csv_path = str(Path(tmp) / "multi.csv")
            logger.export_csv(csv_path)

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 10


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestAuditFullPipeline:
    """End-to-end: hash → log → read → verify → export."""

    def test_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(tmp)

            torch.manual_seed(99)
            waveform = torch.rand(16000) * 2.0 - 1.0
            audio_hash = compute_audio_hash(waveform, 16000)

            assert len(audio_hash) == 64

            rec = AuditRecord(
                model_id="voice-v1",
                watermark_type="classical",
                action="synthesize",
                user_id="user-pipeline",
                watermark_id="wm-pipeline",
                output_hash=audio_hash,
                detection_result=True,
                verification_status=True,
                metadata={"pipeline": "test", "week": 8},
            )
            logger.log(rec)

            records = logger.read_all()
            assert len(records) == 1
            assert records[0].output_hash == audio_hash
            assert records[0].metadata["week"] == 8

            assert logger.verify_integrity() is True

            csv_path = str(Path(tmp) / "pipeline.csv")
            logger.export_csv(csv_path)
            assert Path(csv_path).is_file()
