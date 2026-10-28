from typing import Optional

import torch
import torch.nn.functional as F

from echoself_ai.transformer.model import TransformerLanguageModel


@torch.no_grad()
def generate(
    model: TransformerLanguageModel,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate new tokens autoregressively given a prompt tensor.

    Args:
        model: TransformerLanguageModel instance.
        prompt_tokens: Starting token IDs of shape (batch_size, seq_len).
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (> 0.0). Lower = more deterministic.
        top_k: If set, retain only the top_k most likely tokens for sampling.
        eos_id: Optional token ID indicating end-of-sequence.

    Returns:
        Tensor of shape (batch_size, seq_len + new_tokens) containing the full sequence.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    model.eval()
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        # Crop context to model's maximum context length if necessary
        idx_cond = (
            tokens
            if tokens.size(1) <= model.config.block_size
            else tokens[:, -model.config.block_size :]
        )

        # Forward pass to get logits for all positions
        logits, _ = model(idx_cond)

        # Focus only on the last time step: (batch_size, vocab_size)
        logits = logits[:, -1, :] / temperature

        # Optionally filter logits to top_k candidates
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to running sequence
        tokens = torch.cat((tokens, next_token), dim=1)

        # Early stopping if eos_id provided and generated for all sequences
        if eos_id is not None and (next_token == eos_id).all():
            break

    return tokens
