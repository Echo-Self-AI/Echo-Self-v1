from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters and configuration for a decoder-only Transformer.

    Attributes:
        vocab_size: Number of unique tokens in vocabulary.
        block_size: Maximum context length (sequence length) the model can process.
        n_embd: Latent embedding dimension.
        n_head: Number of attention heads. n_embd must be divisible by n_head.
        n_layer: Number of Transformer blocks.
        dropout: Dropout probability (0.0 <= dropout < 1.0).
        bias: Whether to include bias terms in linear and LayerNorm layers.
    """

    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be at least 1, got {self.vocab_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be at least 1, got {self.block_size}")
        if self.n_embd < 1:
            raise ValueError(f"n_embd must be at least 1, got {self.n_embd}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be at least 1, got {self.n_head}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be at least 1, got {self.n_layer}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {self.dropout}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each individual attention head."""
        return self.n_embd // self.n_head


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Safely select execution device.

    Args:
        prefer_cuda: If True, uses CUDA if available; otherwise falls back to CPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
