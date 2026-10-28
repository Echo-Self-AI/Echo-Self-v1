import math
from typing import Optional
import torch


class CosineWarmupScheduler:
    """Cosine Annealing learning rate schedule with linear warmup.

    Provides smooth learning rate warmup during initial training steps to stabilize
    attention heads and embeddings, followed by a cosine decay down to min_lr.

    Args:
        optimizer: PyTorch optimizer whose learning rate will be scheduled.
        warmup_steps: Number of initial steps to linearly ramp from min_lr to max_lr.
        max_steps: Total training steps.
        max_lr: Peak learning rate reached at the end of warmup.
        min_lr: Final minimum learning rate floor (default: 0.1 * max_lr).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: Optional[float] = None,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if warmup_steps > max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) cannot exceed max_steps ({max_steps})"
            )
        if max_lr <= 0:
            raise ValueError(f"max_lr must be > 0, got {max_lr}")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else 0.1 * max_lr

        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")

        self.current_step = 0

    def get_lr(self, step: int) -> float:
        """Compute the learning rate at a given step."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        # 1) Linear warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * (step / self.warmup_steps)

        # 2) Beyond max_steps, clamp to min_lr
        if step >= self.max_steps:
            return self.min_lr

        # 3) Cosine decay phase
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, step: Optional[int] = None) -> float:
        """Advance the scheduler by one step and apply to optimizer."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr
