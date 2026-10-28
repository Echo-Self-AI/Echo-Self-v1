from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Optional

import torch
from torch import nn

from echoself_ai.transformer.config import TransformerConfig, get_device
from echoself_ai.transformer.model import TransformerLanguageModel


@dataclass
class TrainerConfig:
    """Hyperparameters and operational configuration for training."""

    steps: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 5
    seed: int = 42
    checkpoint_dir: Optional[Path] = None
    device: str = "cpu"  # "cpu", "cuda", or "auto"
    use_cosine_scheduler: bool = False
    warmup_steps: int = 0
    min_lr: Optional[float] = None
    resume_from: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0, got {self.grad_clip}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {self.eval_interval}")
        if self.eval_batches < 1:
            raise ValueError(f"eval_batches must be >= 1, got {self.eval_batches}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr is not None and self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.resume_from is not None:
            self.resume_from = Path(self.resume_from)




def get_batch(
    token_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) sequences."""
    max_idx = len(token_ids) - block_size
    if max_idx < 1:
        raise ValueError(
            f"token_ids length ({len(token_ids)}) must be > block_size ({block_size})"
        )

    starts = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([token_ids[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    """Evaluate average cross-entropy loss over multiple batches without gradient tracking."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    for _ in range(eval_batches):
        x, y = get_batch(token_ids, block_size, batch_size, device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / eval_batches


def save_checkpoint(
    path: Path,
    model: TransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TransformerConfig,
    trainer_config: TrainerConfig,
    vocab_characters: list[str],
    metrics: dict[str, Any],
) -> None:
    """Save training state and model configuration safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "model_config": config.__dict__,
        "trainer_config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in trainer_config.__dict__.items()
        },
        "vocab_characters": vocab_characters,
        "vocab_size": config.vocab_size,
        "metrics": metrics,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    model: Optional[TransformerLanguageModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    """Load model checkpoint safely with weights_only=True compatibility."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # Use weights_only=False to allow loading the full metadata dictionary
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class Trainer:
    """End-to-end training loop for TransformerLanguageModel."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        config: TrainerConfig,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor,
        vocab_characters: list[str],
    ) -> None:
        self.model = model
        self.config = config
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.vocab_characters = vocab_characters

        # Resolve device
        if config.device == "auto":
            self.device = get_device(prefer_cuda=True)
        elif config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Set reproducible seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        from echoself_ai.training.scheduler import CosineWarmupScheduler

        self.scheduler: Optional[CosineWarmupScheduler] = None
        if config.use_cosine_scheduler:
            self.scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=config.warmup_steps,
                max_steps=config.steps,
                max_lr=config.learning_rate,
                min_lr=config.min_lr,
            )

        # History tracking
        self.history: list[dict[str, float]] = []

    def train(self) -> dict[str, Any]:
        """Execute the configured training loop, supporting resuming from checkpoint."""
        self.model.train()
        block_size = self.model.config.block_size
        batch_size = self.config.batch_size

        start_step = 0
        best_val_loss = float("inf")

        # Resume state if configured
        if self.config.resume_from is not None:
            ckpt = load_checkpoint(
                self.config.resume_from,
                device=self.device,
                model=self.model,
                optimizer=self.optimizer,
            )
            start_step = int(ckpt.get("step", 0))
            if "metrics" in ckpt and "best_val_loss" in ckpt["metrics"]:
                best_val_loss = float(ckpt["metrics"]["best_val_loss"])
            elif "metrics" in ckpt and "val_loss" in ckpt["metrics"]:
                best_val_loss = float(ckpt["metrics"]["val_loss"])

            if self.scheduler is not None:
                self.scheduler.current_step = start_step
                new_lr = self.scheduler.get_lr(start_step)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = new_lr

        # Initial metrics
        init_train_loss = evaluate_loss(
            self.model,
            self.train_ids,
            block_size,
            batch_size,
            self.config.eval_batches,
            self.device,
        )
        init_val_loss = evaluate_loss(
            self.model,
            self.val_ids,
            block_size,
            batch_size,
            self.config.eval_batches,
            self.device,
        )

        if best_val_loss == float("inf"):
            best_val_loss = init_val_loss

        for step in range(start_step + 1, self.config.steps + 1):
            x, y = get_batch(self.train_ids, block_size, batch_size, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model(x, targets=y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()


            # Advance learning rate scheduler if enabled
            if self.scheduler is not None:
                self.scheduler.step(step)


            # Periodic evaluation
            if step == 1 or step % self.config.eval_interval == 0 or step == self.config.steps:
                val_loss = evaluate_loss(
                    self.model,
                    self.val_ids,
                    block_size,
                    batch_size,
                    self.config.eval_batches,
                    self.device,
                )
                self.history.append(
                    {
                        "step": float(step),
                        "train_loss": float(loss.item()),
                        "val_loss": float(val_loss),
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.config.checkpoint_dir is not None:
                        save_checkpoint(
                            self.config.checkpoint_dir / "best_model.pt",
                            self.model,
                            self.optimizer,
                            step,
                            self.model.config,
                            self.config,
                            self.vocab_characters,
                            {"step": step, "train_loss": loss.item(), "val_loss": val_loss},
                        )

        # Save final checkpoint if directory configured
        if self.config.checkpoint_dir is not None:
            save_checkpoint(
                self.config.checkpoint_dir / "latest_model.pt",
                self.model,
                self.optimizer,
                self.config.steps,
                self.model.config,
                self.config,
                self.vocab_characters,
                {
                    "step": self.config.steps,
                    "train_loss": self.history[-1]["train_loss"],
                    "val_loss": self.history[-1]["val_loss"],
                    "best_val_loss": best_val_loss,
                },
            )

        return {
            "initial_train_loss": init_train_loss,
            "initial_val_loss": init_val_loss,
            "final_train_loss": self.history[-1]["train_loss"],
            "final_val_loss": self.history[-1]["val_loss"],
            "best_val_loss": best_val_loss,
            "history": self.history,
        }
