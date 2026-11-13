"""
echoself_ai.audit
=================
Append-only audit logging for watermark and voice synthesis events.

Public API
----------
* :class:`AuditRecord` — dataclass representing a single audit event
* :class:`AuditLogger` — appends records to a JSON Lines log file
* :func:`compute_audio_hash` — SHA-256 hex digest of raw PCM bytes
"""

from .logger import AuditLogger, AuditRecord, compute_audio_hash

__all__ = [
    "AuditLogger",
    "AuditRecord",
    "compute_audio_hash",
]
