import math
import pytest
import torch

from echoself_ai.training.scheduler import CosineWarmupScheduler
from echoself_ai.training.trainer import Trainer, TrainerConfig
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel


def test_cosine_warmup_curve():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)

    scheduler = CosineWarmupScheduler(
        optimizer=opt,
        warmup_steps=10,
        max_steps=100,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    # Step 0: min_lr
    assert math.isclose(scheduler.get_lr(0), 1e-4, rel_tol=1e-5)

    # Step 5 (halfway through warmup)
    expected_step_5 = 1e-4 + (1e-3 - 1e-4) * 0.5
    assert math.isclose(scheduler.get_lr(5), expected_step_5, rel_tol=1e-5)

    # Step 10: peak max_lr
    assert math.isclose(scheduler.get_lr(10), 1e-3, rel_tol=1e-5)

    # Step 100: decay down to min_lr
    assert math.isclose(scheduler.get_lr(100), 1e-4, rel_tol=1e-5)

    # Step > 100: stays at min_lr
    assert math.isclose(scheduler.get_lr(150), 1e-4, rel_tol=1e-5)


def test_scheduler_updates_optimizer_param_group():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)

    scheduler = CosineWarmupScheduler(
        optimizer=opt,
        warmup_steps=5,
        max_steps=20,
        max_lr=2e-3,
        min_lr=1e-4,
    )

    lr_after_step = scheduler.step(5)
    assert opt.param_groups[0]["lr"] == lr_after_step
    assert math.isclose(opt.param_groups[0]["lr"], 2e-3, rel_tol=1e-5)


def test_scheduler_invalid_arguments():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1e-3)

    with pytest.raises(ValueError, match="warmup_steps"):
        CosineWarmupScheduler(opt, warmup_steps=-1, max_steps=10, max_lr=1e-3)

    with pytest.raises(ValueError, match="max_steps"):
        CosineWarmupScheduler(opt, warmup_steps=0, max_steps=0, max_lr=1e-3)

    with pytest.raises(ValueError, match="cannot exceed max_steps"):
        CosineWarmupScheduler(opt, warmup_steps=20, max_steps=10, max_lr=1e-3)

    with pytest.raises(ValueError, match="max_lr"):
        CosineWarmupScheduler(opt, warmup_steps=5, max_steps=10, max_lr=0.0)


def test_trainer_with_cosine_scheduler():
    config = TransformerConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = TransformerLanguageModel(config)

    t_config = TrainerConfig(
        steps=10,
        batch_size=2,
        learning_rate=1e-3,
        use_cosine_scheduler=True,
        warmup_steps=2,
        min_lr=1e-4,
        eval_interval=5,
        eval_batches=2,
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
    assert trainer.scheduler is not None
    assert trainer.optimizer.param_groups[0]["lr"] <= 1e-3
    assert results["final_train_loss"] is not None
