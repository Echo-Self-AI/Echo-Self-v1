import math
import torch
from torch import nn
import torch.nn.functional as F

from echoself_ai.transformer.config import TransformerConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention implemented with PyTorch primitives.

    Ensures that for any token at position t, attention can only be paid to
    positions <= t. Future positions are masked with -infinity prior to softmax.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # Combined query, key, value projections in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular boolean mask registered as persistent buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()

        if T > self.causal_mask.size(-1):
            raise ValueError(
                f"Sequence length {T} exceeds maximum configured block_size {self.causal_mask.size(-1)}"
            )

        # Project input to queries, keys, and values
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask: mask positions where causal_mask is False
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted combination of values: (B, n_head, T, head_dim)
        out = weights @ v

        # Re-assemble all head outputs side by side: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection with residual dropout
        return self.resid_dropout(self.c_proj(out))
