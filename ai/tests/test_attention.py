import pytest
import torch

from echoself_ai.transformer.attention import CausalSelfAttention
from echoself_ai.transformer.config import TransformerConfig


def test_attention_output_shape():
    config = TransformerConfig(vocab_size=32, block_size=16, n_embd=32, n_head=4)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 8, 32)
    y = attn(x)

    assert y.shape == (2, 8, 32)


def test_attention_causal_masking_prevents_future_leakage():
    """Verify that changing future tokens does not alter representations of earlier tokens."""
    torch.manual_seed(42)
    config = TransformerConfig(vocab_size=32, block_size=16, n_embd=32, n_head=4, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()

    # Input sequence of length 6
    x1 = torch.randn(1, 6, 32)

    # Clone x1 and mutate only tokens at positions >= 4
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn(1, 2, 32)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Positions 0..3 must match exactly because future tokens (4, 5) are masked out
    torch.testing.assert_close(out1[:, :4, :], out2[:, :4, :], rtol=1e-5, atol=1e-6)

    # Position 4 should differ because token 4 was changed
    assert not torch.allclose(out1[:, 4:, :], out2[:, 4:, :])


def test_attention_gradient_flow():
    config = TransformerConfig(vocab_size=32, block_size=16, n_embd=32, n_head=4)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 6, 32, requires_grad=True)

    y = attn(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for param in attn.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_attention_exceeding_block_size_raises_error():
    config = TransformerConfig(vocab_size=32, block_size=8, n_embd=32, n_head=4)
    attn = CausalSelfAttention(config)
    x = torch.randn(1, 9, 32)

    with pytest.raises(ValueError, match="exceeds maximum configured block_size"):
        attn(x)
