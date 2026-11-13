"""
Append-only audit logger for EchoSelf watermark and synthesis events.

Records are written as JSON Lines (one JSON object per line) to
``ai/data/audit/audit_log.jsonl``.  The log directory is created with
mode 0o700 so only the owning user can read it.

The module deliberately never stores raw audio bytes.  Callers must
convert audio to a hash via :func:`compute_audio_hash` before logging.

Public API
----------
* :class:`AuditRecord` — dataclass for a single audit event
* :class:`AuditLogger` — appends, reads, verifies, and exports records
* :func:`compute_audio_hash` — SHA-256 hex of raw PCM bytes from a waveform
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import struct
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# compute_audio_hash
# ---------------------------------------------------------------------------

def compute_audio_hash(waveform: torch.Tensor, sample_rate: int) -> str:
    """Compute a SHA-256 hex digest of the raw PCM representation of *waveform*.

    The waveform is converted to 16-bit little-endian PCM bytes (the same
    format used by :func:`~echoself_ai.voice.audio.write_wav`) before hashing.
    *sample_rate* is **not** included in the hash — only the raw samples.

    Args:
        waveform: 1-D (or multi-dimensional) float32 tensor.  Values should
                  be in [-1, 1]; values outside that range are clamped before
                  conversion.
        sample_rate: Audio sample rate in Hz.  Accepted but not hashed (kept
                     for a consistent, future-proof API).

    Returns:
        64-character lowercase hexadecimal SHA-256 digest string.
    """
    flat = waveform.detach().cpu().float().flatten()
    clamped = torch.clamp(flat, -1.0, 1.0)
    int16_vals = (clamped * 32767.0).to(torch.int16).tolist()
    raw_bytes = struct.pack(f"<{len(int16_vals)}h", *int16_vals)
    return hashlib.sha256(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({"synthesize", "detect", "verify", "enroll", "revoke"})


@dataclass
class AuditRecord:
    """A single immutable audit event.

    Attributes:
        event_id: UUID string uniquely identifying this event.
        user_id: Opaque identifier of the requesting user (may be None).
        timestamp: ISO-8601 UTC timestamp (e.g. ``'2026-11-13T07:00:00+00:00'``).
        model_id: Identifier of the model that produced or processed the audio.
        watermark_type: One of ``'classical'``, ``'learned'``, ``'none'``,
                        ``'unsupported'``.
        watermark_id: Watermark identifier (may be None for detection-only events).
        output_hash: SHA-256 hex of output audio bytes, or None if not applicable.
        detection_result: Whether a watermark was detected (None if not a detection
                          event).
        verification_status: Whether integrity verification passed (None if not
                             applicable).
        action: One of ``'synthesize'``, ``'detect'``, ``'verify'``,
                ``'enroll'``, ``'revoke'``.
        metadata: Arbitrary key/value pairs for additional context.
    """

    model_id: str
    watermark_type: str
    action: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    watermark_id: Optional[str] = None
    output_hash: Optional[str] = None
    detection_result: Optional[bool] = None
    verification_status: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(VALID_ACTIONS)}, got {self.action!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Reconstruct an :class:`AuditRecord` from a dictionary.

        Unknown keys are silently ignored so that future fields added to the
        schema do not break older readers.
        """
        known_fields = {
            "event_id",
            "user_id",
            "timestamp",
            "model_id",
            "watermark_type",
            "watermark_id",
            "output_hash",
            "detection_result",
            "verification_status",
            "action",
            "metadata",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only audit logger backed by a JSON Lines file.

    The log directory is created with mode ``0o700`` (owner read/write/execute
    only) on first use.  Raw audio bytes are **never** stored; use
    :func:`compute_audio_hash` to obtain a safe hash to log instead.

    Args:
        log_path: Path to the ``.jsonl`` file.  Defaults to
                  ``{repo_ai_dir}/data/audit/audit_log.jsonl``.

    Example::

        logger = AuditLogger()
        record = AuditRecord(
            model_id="voice-v1",
            watermark_type="classical",
            action="synthesize",
            output_hash=compute_audio_hash(waveform, 16000),
        )
        logger.log(record)
        all_records = logger.read_all()
        ok = logger.verify_integrity()
        logger.export_csv("/tmp/audit.csv")
    """

    # Default path relative to this file: ai/src/echoself_ai/audit/logger.py
    # → go up 4 levels to reach ai/, then data/audit/audit_log.jsonl
    _DEFAULT_RELATIVE_PATH = Path("data") / "audit" / "audit_log.jsonl"

    def __init__(self, log_path: Optional[str] = None) -> None:
        if log_path is not None:
            self._log_path = Path(log_path)
        else:
            # Navigate: logger.py → audit → echoself_ai → src → ai
            this_file = Path(__file__).resolve()
            ai_dir = this_file.parents[3]
            self._log_path = ai_dir / self._DEFAULT_RELATIVE_PATH

        self._ensure_log_dir()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_log_dir(self) -> None:
        """Create the log directory with restricted permissions if needed."""
        log_dir = self._log_path.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            # Set to 0o700: owner rwx only
            os.chmod(log_dir, stat.S_IRWXU)
        elif not log_dir.is_dir():
            raise NotADirectoryError(
                f"Log path parent exists but is not a directory: {log_dir}"
            )

    def _serialize(self, record: AuditRecord) -> str:
        """Serialize *record* to a compact JSON string (no trailing newline)."""
        return json.dumps(record.to_dict(), separators=(",", ":"), ensure_ascii=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, record: AuditRecord) -> None:
        """Append *record* as a single JSON line to the log file.

        The write is performed in append mode so concurrent writers will not
        corrupt existing lines (last-line atomicity is guaranteed on POSIX for
        writes smaller than PIPE_BUF, but no cross-process locking is used
        here).

        Args:
            record: :class:`AuditRecord` to persist.
        """
        self._ensure_log_dir()
        line = self._serialize(record) + "\n"
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def read_all(self) -> List[AuditRecord]:
        """Read and return all :class:`AuditRecord` objects from the log.

        Empty or missing log files return an empty list.

        Returns:
            List of :class:`AuditRecord` in insertion order.

        Raises:
            :class:`json.JSONDecodeError` if a line is not valid JSON.  Call
            :meth:`verify_integrity` first if you want a non-raising check.
        """
        if not self._log_path.is_file():
            return []

        records: List[AuditRecord] = []
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                records.append(AuditRecord.from_dict(data))
        return records

    def verify_integrity(self) -> bool:
        """Verify that every line in the log file is valid JSON.

        This is a lightweight tamper-detection check: any direct text edit
        that breaks JSON syntax (injected characters, truncated records, etc.)
        will be caught here.

        Returns:
            ``True`` if all lines parse successfully; ``False`` if any line
            fails JSON parsing.
        """
        if not self._log_path.is_file():
            return True  # Empty / missing log is trivially valid

        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    json.loads(stripped)  # raises ValueError / json.JSONDecodeError on bad JSON
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def export_csv(self, path: str) -> None:
        """Export all log records to a CSV file at *path*.

        The CSV header is derived from the fields of :class:`AuditRecord` in
        the order they are defined.  The ``metadata`` column is serialised as a
        compact JSON string.

        Args:
            path: Destination file path (created or overwritten).
        """
        records = self.read_all()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "event_id",
            "user_id",
            "timestamp",
            "model_id",
            "watermark_type",
            "watermark_id",
            "output_hash",
            "detection_result",
            "verification_status",
            "action",
            "metadata",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = rec.to_dict()
                # Serialise the nested dict as a JSON string for the CSV cell
                row["metadata"] = json.dumps(row.get("metadata", {}), separators=(",", ":"))
                # Ensure all expected keys are present
                writer.writerow({k: row.get(k, "") for k in fieldnames})
