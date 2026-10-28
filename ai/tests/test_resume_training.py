from pathlib import Path
import pytest
import torch

from echoself_ai.training.trainer import Trainer, TrainerConfig
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel


def test_resume_training_from_checkpoint(tmp_path: Path):
    config = TransformerConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)
    ckpt_dir = tmp_path / "checkpoints"

    train_ids = torch.randint(0, 10, (100,))
    val_ids = torch.randint(0, 10, (50,))

    # 1) Initial training run for 10 steps
    t_config1 = TrainerConfig(
        steps=10,
        batch_size=2,
        learning_rate=1e-3,
        eval_interval=5,
        checkpoint_dir=ckpt_dir,
        device="cpu",
    )
    trainer1 = Trainer(
        model,
        t_config1,
        train_ids,
        val_ids,
        vocab_characters=[str(i) for i in range(10)],
    )
    res1 = trainer1.train()
    assert (ckpt_dir / "latest_model.pt").exists()

    # 2) Resume training for up to 20 steps using fresh model instance
    fresh_model = TransformerLanguageModel(config)
    t_config2 = TrainerConfig(
        steps=20,
        batch_size=2,
        learning_rate=1e-3,
        eval_interval=5,
        checkpoint_dir=ckpt_dir,
        resume_from=ckpt_dir / "latest_model.pt",
        device="cpu",
    )
    trainer2 = Trainer(
        fresh_model,
        t_config2,
        train_ids,
        val_ids,
        vocab_characters=[str(i) for i in range(10)],
    )
    res2 = trainer2.train()

    # Verify history in second run starts after step 10
    steps_run = [entry["step"] for entry in res2["history"]]
    assert all(s > 10 for s in steps_run)
    assert steps_run[-1] == 20.0


def test_resume_from_nonexistent_checkpoint_raises(tmp_path: Path):
    config = TransformerConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)

    t_config = TrainerConfig(
        steps=10,
        resume_from=tmp_path / "does_not_exist.pt",
        device="cpu",
    )
    trainer = Trainer(
        model,
        t_config,
        train_ids=torch.randint(0, 10, (50,)),
        val_ids=torch.randint(0, 10, (20,)),
        vocab_characters=["a", "b"],
    )

    with pytest.raises(FileNotFoundError):
        trainer.train()
