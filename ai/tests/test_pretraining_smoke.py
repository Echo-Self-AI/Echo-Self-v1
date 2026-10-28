"""Smoke test for language model pretraining components before full training execution."""
from pathlib import Path
import pytest
import torch

from echoself_ai.bpe_tokenizer import BPETokenizer
from echoself_ai.data_manager import fetch_and_prepare_dataset
from echoself_ai.training.trainer import get_batch, load_checkpoint, save_checkpoint, TrainerConfig
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel


def test_pretraining_smoke_pipeline(tmp_path: Path):
    # 1. Fetch/verify dataset
    data_path, manifest = fetch_and_prepare_dataset("tinyshakespeare", target_dir=tmp_path / "data")
    assert data_path.is_file()
    assert manifest.num_characters > 100000

    text = data_path.read_text(encoding="utf-8")[:1500]  # Smoke slice

    # 2. Train subword tokenizer on training split
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    bpe = BPETokenizer()
    bpe.train(train_text, target_vocab_size=80)
    assert bpe.vocab_size == 80

    train_ids = torch.tensor(bpe.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(bpe.encode(val_text), dtype=torch.long)
    assert (train_ids >= 0).all() and (train_ids < bpe.vocab_size).all()
    assert (val_ids >= 0).all() and (val_ids < bpe.vocab_size).all()

    # 3. Verify batch shape and next-token target shift
    batch_size = 4
    block_size = 16
    device = torch.device("cpu")
    x, y = get_batch(train_ids, block_size=block_size, batch_size=batch_size, device=device)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # 4. Initialize Tiny Transformer
    cfg = TransformerConfig(
        vocab_size=bpe.vocab_size,
        block_size=block_size,
        n_embd=32,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )
    model = TransformerLanguageModel(cfg).to(device)

    # 5. Forward pass and loss calculation
    logits, loss = model(x, targets=y)
    assert logits.shape == (batch_size, block_size, bpe.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    initial_loss_val = loss.item()
    assert initial_loss_val > 0.0

    # 6. Backward pass and gradient confirmation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            if p.grad.abs().sum() > 0:
                has_grad = True
    assert has_grad, "Expected nonzero gradients in model parameters"

    # 7. Optimizer step
    optimizer.step()
    post_step_logits, _ = model(x)

    # 8. Save and reload checkpoint
    ckpt_path = tmp_path / "smoke_model.pt"
    t_cfg = TrainerConfig(steps=1, batch_size=batch_size)
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=1,
        config=cfg,
        trainer_config=t_cfg,
        vocab_characters=bpe.vocab,
        metrics={"loss": initial_loss_val},
    )
    assert ckpt_path.is_file()
    assert ckpt_path.with_suffix(".pt.sha256").is_file()

    reloaded_model = TransformerLanguageModel(cfg)
    load_checkpoint(ckpt_path, device=device, model=reloaded_model)
    reloaded_logits, _ = reloaded_model(x)
    assert torch.allclose(post_step_logits, reloaded_logits, atol=1e-5)

