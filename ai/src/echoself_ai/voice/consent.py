from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Optional


@dataclass
class VoiceConsentRecord:
    """Formal verifiable record of user authorization for voice processing."""

    user_id: str
    speaker_id: str
    consent_granted: bool
    scope: str  # "synthesis", "cloning", or "training"
    granted_at: str
    verification_hash: str
    revoked: bool = False
    revoked_at: Optional[str] = None

    def is_active_for(self, scope: str) -> bool:
        """Check if consent is valid, active, and covers the requested operation scope."""
        if not self.consent_granted or self.revoked:
            return False
        if self.scope == "all" or self.scope == scope:
            return True
        return False


class VoiceConsentManager:
    """Manages explicit user authorization, gating, and data deletion for Voice AI."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("data/consent")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, VoiceConsentRecord] = {}
        self._load_all()

    def _get_key(self, user_id: str, speaker_id: str) -> str:
        return f"{user_id}_{speaker_id}"

    def _load_all(self) -> None:
        """Load registered consent records from storage."""
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = VoiceConsentRecord(**data)
                self.records[self._get_key(record.user_id, record.speaker_id)] = record
            except Exception:
                continue

    def register_consent(
        self,
        user_id: str,
        speaker_id: str,
        scope: str = "synthesis",
        authorization_text: str = "I authorize EchoSelf to process my voice samples.",
    ) -> VoiceConsentRecord:
        """Record explicit consent with cryptographic verification hash."""
        now = datetime.now(timezone.utc).isoformat()
        hasher = hashlib.sha256()
        hasher.update(f"{user_id}:{speaker_id}:{scope}:{authorization_text}:{now}".encode("utf-8"))
        verification_hash = hasher.hexdigest()

        record = VoiceConsentRecord(
            user_id=user_id,
            speaker_id=speaker_id,
            consent_granted=True,
            scope=scope,
            granted_at=now,
            verification_hash=verification_hash,
            revoked=False,
        )

        key = self._get_key(user_id, speaker_id)
        self.records[key] = record

        record_file = self.storage_dir / f"{key}.json"
        record_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

        return record

    def check_consent(self, user_id: str, speaker_id: str, scope: str = "synthesis") -> bool:
        """Query if active authorized consent exists."""
        key = self._get_key(user_id, speaker_id)
        record = self.records.get(key)
        if record is None:
            return False
        return record.is_active_for(scope)

    def enforce_gate(self, user_id: str, speaker_id: str, scope: str = "synthesis") -> None:
        """Enforce strict consent gate, raising PermissionError if authorization is absent."""
        if not self.check_consent(user_id, speaker_id, scope):
            raise PermissionError(
                f"Voice operation '{scope}' denied: No active consent for user '{user_id}' "
                f"and speaker '{speaker_id}'."
            )

    def revoke_consent(self, user_id: str, speaker_id: str) -> None:
        """Revoke previously granted consent."""
        key = self._get_key(user_id, speaker_id)
        record = self.records.get(key)
        if record is None:
            raise KeyError(f"No consent record found for {key}")

        record.revoked = True
        record.revoked_at = datetime.now(timezone.utc).isoformat()

        record_file = self.storage_dir / f"{key}.json"
        record_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

    def purge_speaker_data(
        self,
        user_id: str,
        speaker_id: str,
        data_directory: Path,
    ) -> int:
        """Purge all stored voice audio files, embeddings, and caches for a speaker upon revocation.

        Returns:
            Number of files removed.
        """
        self.enforce_revoked(user_id, speaker_id)

        data_directory = Path(data_directory)
        if not data_directory.exists():
            return 0

        target_prefix = f"{speaker_id}_"
        removed_count = 0

        for file_path in list(data_directory.glob(f"{target_prefix}*")):
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                removed_count += 1

        return removed_count

    def enforce_revoked(self, user_id: str, speaker_id: str) -> None:
        """Confirm that consent was revoked before allowing data purge."""
        key = self._get_key(user_id, speaker_id)
        record = self.records.get(key)
        if record is None or not record.revoked:
            raise ValueError(f"Cannot purge data: consent for {key} has not been revoked.")
