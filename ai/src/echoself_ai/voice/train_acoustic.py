import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from echoself_ai.voice.acoustic_model import AcousticModel, AcousticModelConfig
from echoself_ai.voice.dataset import SpeechDataset, SpeechVocab, collate_speech_batch
from echoself_ai.voice.vocoder import GriffinLimVocoder


def save_acoustic_checkpoint(
    path: str | Path,
    model: AcousticModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    val_loss: float,
    config: AcousticModelConfig,
    vocab: SpeechVocab,
) -> Path:
    """Save acoustic model checkpoint with SHA-256 sidecar checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "n_mels": config.n_mels,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_speakers": config.num_speakers,
            "upsample_factor": config.upsample_factor,
            "dropout": config.dropout,
        },
        "vocab_size": vocab.size,
    }

    torch.save(state, path)

    # Compute and save SHA-256 checksum sidecar
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    digest = hasher.hexdigest()

    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")

    return path


def load_acoustic_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    model: Optional[AcousticModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Load acoustic model checkpoint safely onto specified device."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def evaluate_acoustic(
    model: AcousticModel,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate mean L1 mel reconstruction loss on validation split."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            token_ids = batch["token_ids"].to(device)
            target_mel = batch["target_mel"].to(device)
            speaker_ids = batch["speaker_id"].to(device)

            predicted_mel, loss = model(
                token_ids=token_ids,
                speaker_id=speaker_ids,
                target_mel=target_mel,
            )

            if loss is not None:
                total_loss += loss.item() * token_ids.size(0)
                count += token_ids.size(0)

    return total_loss / max(1, count)


def run_acoustic_training(
    metadata_path: str | Path,
    wavs_dir: str | Path,
    steps: int = 200,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    upsample_factor: int = 8,
    device_str: str = "auto",
    checkpoint_dir: str | Path = "checkpoints/acoustic_real",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute acoustic model training run on speech dataset."""
    torch.manual_seed(seed)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    vocab = SpeechVocab()
    train_dataset = SpeechDataset(
        metadata_path=metadata_path,
        wavs_dir=wavs_dir,
        vocab=vocab,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    val_dataset = SpeechDataset(
        metadata_path=metadata_path,
        wavs_dir=wavs_dir,
        vocab=vocab,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_speech_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_batch,
    )

    model_config = AcousticModelConfig(
        vocab_size=vocab.size,
        n_mels=80,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        upsample_factor=upsample_factor,
        dropout=0.1,
    )
    model = AcousticModel(model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initial_val_loss = evaluate_acoustic(model, val_loader, device)

    print("=" * 60)
    print("EchoSelf AI — Acoustic Model Pretraining")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Train utterances: {len(train_dataset)} | Val utterances: {len(val_dataset)}")
    print(f"Vocab size: {vocab.size} | Hidden dim: {hidden_dim} | Layers: {num_layers}")
    print(f"Upsample factor: {upsample_factor} | Steps: {steps} | Batch size: {batch_size}")
    print(f"Initial val loss: {initial_val_loss:.4f}")
    print("-" * 60)

    best_val_loss = initial_val_loss
    step = 0
    epoch = 0
    start_time = time.time()
    train_losses: list[float] = []

    model.train()
    while step < steps:
        epoch += 1
        for batch in train_loader:
            if step >= steps:
                break

            token_ids = batch["token_ids"].to(device)
            target_mel = batch["target_mel"].to(device)
            speaker_ids = batch["speaker_id"].to(device)

            optimizer.zero_grad()
            _, loss = model(
                token_ids=token_ids,
                speaker_id=speaker_ids,
                target_mel=target_mel,
            )

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            step += 1

            if step % 50 == 0 or step == steps:
                val_loss = evaluate_acoustic(model, val_loader, device)
                model.train()
                elapsed = time.time() - start_time
                curr_train_loss = train_losses[-1] if train_losses else 0.0
                print(f"Step {step:4d}/{steps} | Train L1: {curr_train_loss:.4f} | Val L1: {val_loss:.4f} | {elapsed:.1f}s")

                if val_loss < best_val_loss or step == steps:
                    best_val_loss = min(best_val_loss, val_loss)
                    save_acoustic_checkpoint(
                        checkpoint_dir / "best_model.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=step,
                        val_loss=val_loss,
                        config=model_config,
                        vocab=vocab,
                    )

    elapsed_total = time.time() - start_time
    final_train_loss = train_losses[-1] if train_losses else 0.0
    final_val_loss = evaluate_acoustic(model, val_loader, device)

    save_acoustic_checkpoint(
        checkpoint_dir / "latest_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        val_loss=final_val_loss,
        config=model_config,
        vocab=vocab,
    )

    print("-" * 60)
    print(f"Completed {step} steps in {elapsed_total:.2f}s")
    print(f"Initial val loss: {initial_val_loss:.4f}")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final val loss:   {final_val_loss:.4f}")
    print(f"Best val loss:    {best_val_loss:.4f}")
    print("=" * 60)

    # Verification: reload checkpoint and synthesize a sample utterance
    eval_model = AcousticModel(model_config).to(device)
    load_acoustic_checkpoint(checkpoint_dir / "best_model.pt", device=device, model=eval_model)
    eval_model.eval()

    test_prompt = "Printing is the art of multiplying books."
    encoded = vocab.encode(test_prompt)
    prompt_tensor = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        predicted_mel, _ = eval_model(prompt_tensor)

    # Reconstruct audio via Griffin-Lim vocoder
    vocoder = GriffinLimVocoder(sample_rate=16000, n_mels=80, n_iters=16)
    audio = vocoder(predicted_mel.cpu())

    results = {
        "vocab_size": vocab.size,
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "total_steps": step,
        "elapsed_seconds": elapsed_total,
        "reconstructed_samples": audio.size(1),
        "reconstructed_finite": bool(torch.isfinite(audio).all()),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoSelf AI Acoustic Model Training")
    parser.add_argument("--metadata", type=str, default="data/raw/ljspeech/metadata.csv")
    parser.add_argument("--wavs-dir", type=str, default="data/raw/ljspeech/wavs")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--upsample-factor", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/acoustic_real")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_acoustic_training(
        metadata_path=args.metadata,
        wavs_dir=args.wavs_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        upsample_factor=args.upsample_factor,
        device_str=args.device,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
