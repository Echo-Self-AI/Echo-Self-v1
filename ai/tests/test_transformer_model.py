import pytest
import torch

from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import (
    FeedForward,
    TransformerBlock,
    TransformerLanguageModel,
)


def test_feedforward_shape():
    config = TransformerConfig(vocab_size=32, n_embd=32, n_head=4)
    ff = FeedForward(config)
    x = torch.randn(2, 5, 32)
    y = ff(x)
    assert y.shape == (2, 5, 32)


def test_transformer_block_shape():
    config = TransformerConfig(vocab_size=32, block_size=16, n_embd=32, n_head=4)
    block = TransformerBlock(config)
    x = torch.randn(2, 6, 32)
    y = block(x)
    assert y.shape == (2, 6, 32)


def test_model_forward_without_targets():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    idx = torch.randint(0, 20, (2, 8))

    logits, loss = model(idx)
    assert logits.shape == (2, 8, 20)
    assert loss is None


def test_model_forward_with_targets_computes_loss():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    idx = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 8, 20)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0


def test_model_backward_computes_gradients():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    idx = torch.randint(0, 20, (2, 8))
    targets = torch.randint(0, 20, (2, 8))

    _, loss = model(idx, targets=targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"


def test_model_weight_tying():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_model_exceeding_block_size_raises_error():
    config = TransformerConfig(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    idx = torch.randint(0, 20, (1, 9))

    with pytest.raises(ValueError, match="Cannot forward sequence"):
        model(idx)


def test_full_model_causality():
    """Verify that predictions at position t only depend on tokens at positions <= t."""
    torch.manual_seed(1337)
    config = TransformerConfig(
        vocab_size=25, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0
    )
    model = TransformerLanguageModel(config)
    model.eval()

    seq1 = torch.tensor([[1, 2, 3, 4, 5, 6]])
    seq2 = torch.tensor([[1, 2, 3, 4, 9, 10]])  # Differ at positions 4 and 5

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)

    # Logits at positions 0..3 must match exactly
    torch.testing.assert_close(logits1[:, :4, :], logits2[:, :4, :], rtol=1e-5, atol=1e-6)

    # Logits at positions 4..5 should differ
    assert not torch.allclose(logits1[:, 4:, :], logits2[:, 4:, :])
