from pathlib import Path
import pytest
import torch

from echoself_ai.security.integrity import verify_checkpoint_integrity
from echoself_ai.voice.audio import save_wav
from echoself_ai.voice.train_acoustic import (
    load_acoustic_checkpoint,
    run_acoustic_training,
)
from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig


def test_acoustic_training_smoke(tmp_path: Path):
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    # Generate 4 short synthetic audio clips
    sr = 16000
    t = torch.linspace(0, 0.5, int(sr * 0.5))
    wav = 0.3 * torch.sin(2 * torch.pi * 300.0 * t)

    meta_lines: list[str] = []
    for i in range(4):
        clip_id = f"test_{i}"
        save_wav(wavs_dir / f"{clip_id}.wav", wav, sr)
        meta_lines.append(f"{clip_id}|test utterance {i}|test utterance {i}\n")

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("".join(meta_lines), encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"

    results = run_acoustic_training(
        metadata_path=metadata_path,
        wavs_dir=wavs_dir,
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        hidden_dim=16,
        num_layers=2,
        upsample_factor=4,
        device_str="cpu",
        checkpoint_dir=ckpt_dir,
        val_ratio=0.5,
    )

    assert results["total_steps"] == 2
    assert results["final_train_loss"] > 0.0
    assert results["final_val_loss"] > 0.0
    assert results["reconstructed_finite"] is True
    assert results["reconstructed_samples"] > 0

    # Verify checkpoint and sha256 sidecar
    best_ckpt = ckpt_dir / "best_model.pt"
    assert best_ckpt.is_file()
    assert (ckpt_dir / "best_model.pt.sha256").is_file()
    assert verify_checkpoint_integrity(best_ckpt) is True

    # Reload checkpoint
    cfg = AcousticModelConfig(vocab_size=results["vocab_size"], n_mels=80, hidden_dim=16, num_layers=2, upsample_factor=4)
    model = AcousticModel(cfg)
    ckpt_data = load_acoustic_checkpoint(best_ckpt, device="cpu", model=model)
    assert "model_state_dict" in ckpt_data
    assert ckpt_data["step"] <= 2
