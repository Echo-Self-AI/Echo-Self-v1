"""Unit tests for model checkpoint cryptographic integrity and tamper verification."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from echoself_ai.security.integrity import (
    CheckpointIntegrityError,
    compute_file_sha256,
    load_secure_checkpoint,
    save_secure_checkpoint,
    verify_checkpoint_integrity,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_compute_file_sha256_and_non_existent(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("EchoSelf AI Checkpoint", encoding="utf-8")

    digest = compute_file_sha256(f)
    assert isinstance(digest, str)
    assert len(digest) == 64

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "does_not_exist.bin")


def test_save_and_load_secure_checkpoint(tmp_path: Path):
    model = TinyModel()
    ckpt_path = tmp_path / "tiny_model.pt"

    meta = save_secure_checkpoint(
        state_or_model=model,
        target_path=ckpt_path,
        architecture="TinyModel",
        extra_meta={"step": 100},
    )

    assert ckpt_path.is_file()
    sidecar_sha = tmp_path / "tiny_model.pt.sha256"
    assert sidecar_sha.is_file()
    assert meta.sha256_hash in sidecar_sha.read_text()

    sidecar_json = tmp_path / "tiny_model.pt.meta.json"
    assert sidecar_json.is_file()
    assert "TinyModel" in sidecar_json.read_text()

    # Load successfully with verified integrity
    loaded_state = load_secure_checkpoint(ckpt_path, map_location="cpu")
    assert "fc.weight" in loaded_state
    assert "fc.bias" in loaded_state

    # Verify tensor equality
    orig_state = model.state_dict()
    assert torch.allclose(loaded_state["fc.weight"], orig_state["fc.weight"])


def test_tampered_checkpoint_rejection(tmp_path: Path):
    model = TinyModel()
    ckpt_path = tmp_path / "tamper_test.pt"

    save_secure_checkpoint(model, ckpt_path)

    # Verify normal integrity first
    assert verify_checkpoint_integrity(ckpt_path) is True

    # Tamper with the checkpoint file (modify a byte)
    with open(ckpt_path, "r+b") as f:
        f.seek(10)
        byte = f.read(1)
        f.seek(10)
        # Flip bit
        f.write(bytes([byte[0] ^ 0xFF]))

    # Tampered verification should raise CheckpointIntegrityError
    with pytest.raises(CheckpointIntegrityError) as exc_info:
        verify_checkpoint_integrity(ckpt_path)
    assert "Integrity check failed" in str(exc_info.value)

    # Secure loading should abort and raise CheckpointIntegrityError
    with pytest.raises(CheckpointIntegrityError):
        load_secure_checkpoint(ckpt_path)


def test_missing_sidecar_behavior(tmp_path: Path):
    ckpt_path = tmp_path / "no_sidecar.pt"
    torch.save({"dummy": torch.tensor([1, 2, 3])}, ckpt_path)

    # Without sidecar and without expected_sha256, verification fails safely
    with pytest.raises(CheckpointIntegrityError) as exc_info:
        verify_checkpoint_integrity(ckpt_path)
    assert "No expected hash provided" in str(exc_info.value)

    # If explicit hash is provided, verification succeeds
    correct_hash = compute_file_sha256(ckpt_path)
    assert verify_checkpoint_integrity(ckpt_path, expected_sha256=correct_hash) is True


def test_existing_repo_checkpoints_integrity():
    repo_root = Path(__file__).resolve().parent.parent.parent
    checkpoints_dir = repo_root / "ai" / "checkpoints"
    
    sha256_files = list(checkpoints_dir.glob("**/*.sha256"))
    assert len(sha256_files) > 0, "Expected at least one .sha256 sidecar in checkpoints"

    for sidecar in sha256_files:
        pt_file = sidecar.with_suffix("")  # strips .sha256
        assert pt_file.is_file(), f"Target checkpoint {pt_file} missing for sidecar {sidecar}"
        assert verify_checkpoint_integrity(pt_file) is True

