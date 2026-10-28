from pathlib import Path
import pytest
import torch

from echoself_ai.training.trainer import (
    Trainer,
    TrainerConfig,
    evaluate_loss,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel


def test_trainer_config_validation():
    with pytest.raises(ValueError, match="steps"):
        TrainerConfig(steps=0)

    with pytest.raises(ValueError, match="batch_size"):
        TrainerConfig(batch_size=0)

    with pytest.raises(ValueError, match="learning_rate"):
        TrainerConfig(learning_rate=-0.01)

    with pytest.raises(ValueError, match="weight_decay"):
        TrainerConfig(weight_decay=-1.0)

    with pytest.raises(ValueError, match="grad_clip"):
        TrainerConfig(grad_clip=-1.0)


def test_get_batch_shapes():
    tokens = torch.arange(100, dtype=torch.long)
    x, y = get_batch(tokens, block_size=16, batch_size=4, device=torch.device("cpu"))

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y should be shifted by 1 relative to x
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_insufficient_tokens_raises():
    tokens = torch.arange(5, dtype=torch.long)
    with pytest.raises(ValueError, match="must be > block_size"):
        get_batch(tokens, block_size=8, batch_size=2, device=torch.device("cpu"))


def test_evaluate_loss_preserves_training_state():
    config = TransformerConfig(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)
    model.train()
    tokens = torch.randint(0, 20, (100,))

    loss = evaluate_loss(
        model,
        tokens,
        block_size=8,
        batch_size=2,
        eval_batches=3,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0.0
    assert model.training is True  # Preserved original state


def test_save_and_load_checkpoint(tmp_path: Path):
    config = TransformerConfig(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t_config = TrainerConfig(steps=10)
    ckpt_file = tmp_path / "checkpoint.pt"

    save_checkpoint(
        ckpt_file,
        model,
        optimizer,
        step=5,
        config=config,
        trainer_config=t_config,
        vocab_characters=["a", "b"],
        metrics={"loss": 2.5},
    )

    assert ckpt_file.exists()

    # Create fresh model and verify loaded weights match
    loaded_model = TransformerLanguageModel(config)
    loaded_opt = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
    info = load_checkpoint(ckpt_file, model=loaded_model, optimizer=loaded_opt)

    assert info["step"] == 5
    assert info["vocab_characters"] == ["a", "b"]
    assert info["metrics"]["loss"] == 2.5

    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)


def test_trainer_smoke_run(tmp_path: Path):
    # Quick 10-step train on synthetic data
    config = TransformerConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)

    t_config = TrainerConfig(
        steps=10,
        batch_size=2,
        learning_rate=1e-2,
        eval_interval=5,
        eval_batches=2,
        checkpoint_dir=tmp_path / "ckpts",
        device="cpu",
    )

    train_ids = torch.randint(0, 10, (100,))
    val_ids = torch.randint(0, 10, (50,))

    trainer = Trainer(
        model,
        t_config,
        train_ids,
        val_ids,
        vocab_characters=[str(i) for i in range(10)],
    )

    results = trainer.train()

    assert "initial_train_loss" in results
    assert "final_train_loss" in results
    assert len(trainer.history) > 0
    assert (tmp_path / "ckpts" / "latest_model.pt").exists()
