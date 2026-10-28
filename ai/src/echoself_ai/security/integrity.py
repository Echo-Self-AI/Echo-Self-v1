"""Cryptographic integrity verification and safe deserialization for PyTorch checkpoints."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch


class CheckpointIntegrityError(Exception):
    """Raised when a checkpoint fails SHA-256 cryptographic integrity or tamper checks."""
    pass


@dataclass
class CheckpointMetadata:
    """Metadata describing a saved model checkpoint and its cryptographic hash."""

    checkpoint_path: str
    sha256_hash: str
    file_size_bytes: int
    architecture: str
    parameters_count: int
    created_at_utc: Optional[str] = None


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 checksum of a file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint_integrity(
    checkpoint_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> bool:
    """Verify that a checkpoint matches its expected SHA-256 or sidecar .sha256 file.
    
    Returns True if valid, raises CheckpointIntegrityError if corrupted or tampered.
    """
    p = Path(checkpoint_path)
    if not p.is_file():
        raise CheckpointIntegrityError(f"Checkpoint file does not exist: {p}")

    computed_hash = compute_file_sha256(p)

    target_hash = expected_sha256
    if not target_hash:
        sidecar = p.with_suffix(p.suffix + ".sha256")
        if sidecar.is_file():
            target_hash = sidecar.read_text(encoding="utf-8").strip().split()[0]
        else:
            raise CheckpointIntegrityError(
                f"No expected hash provided and sidecar checksum '{sidecar.name}' not found."
            )

    if computed_hash.lower() != target_hash.lower():
        raise CheckpointIntegrityError(
            f"Integrity check failed for {p.name}! "
            f"Expected SHA-256: {target_hash.lower()}, Computed: {computed_hash.lower()}"
        )

    return True


def save_secure_checkpoint(
    state_or_model: Union[torch.nn.Module, dict[str, Any]],
    target_path: Union[str, Path],
    architecture: str = "custom",
    extra_meta: Optional[dict[str, Any]] = None,
) -> CheckpointMetadata:
    """Serialize weights safely, compute SHA-256 digest, and generate .sha256 sidecar."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(state_or_model, torch.nn.Module):
        state_dict = state_or_model.state_dict()
        param_count = sum(p.numel() for p in state_or_model.parameters())
    else:
        state_dict = state_or_model
        param_count = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))

    # Save torch state dict
    torch.save(state_dict, target)

    # Compute SHA-256
    sha256_digest = compute_file_sha256(target)

    # Write sidecar .sha256 file
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{sha256_digest}  {target.name}\n", encoding="utf-8")

    meta = CheckpointMetadata(
        checkpoint_path=str(target.resolve()),
        sha256_hash=sha256_digest,
        file_size_bytes=target.stat().st_size,
        architecture=architecture,
        parameters_count=param_count,
    )

    # Write JSON metadata sidecar
    json_sidecar = target.with_suffix(target.suffix + ".meta.json")
    meta_dict = asdict(meta)
    if extra_meta:
        meta_dict["extra"] = extra_meta
    json_sidecar.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    return meta


def load_secure_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    expected_sha256: Optional[str] = None,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Verify cryptographic integrity, then load state dict strictly with weights_only=True."""
    p = Path(checkpoint_path)

    if verify_hash:
        verify_checkpoint_integrity(p, expected_sha256=expected_sha256)

    # Safely load weights only - prevents arbitrary pickled code execution
    state_dict = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(state_dict, dict):
        raise CheckpointIntegrityError(
            f"Checkpoint payload must be a dictionary state_dict, got {type(state_dict).__name__}"
        )

    return state_dict
