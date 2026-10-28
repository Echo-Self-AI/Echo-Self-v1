import pytest
import torch

from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.generation import generate
from echoself_ai.transformer.model import TransformerLanguageModel


def test_generation_shape():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1, 2, 3]])

    out = generate(model, prompt, max_new_tokens=5)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], prompt)


def test_generation_method_on_model():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1, 2]])

    out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 6)
    assert torch.equal(out[:, :2], prompt)


def test_generation_exceeding_block_size_slides_window():
    """Generating past block_size should crop context gracefully and continue."""
    config = TransformerConfig(vocab_size=20, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    # Total tokens = 5 + 10 = 15 > block_size 8
    out = model.generate(prompt, max_new_tokens=10)
    assert out.shape == (1, 15)


def test_generation_invalid_parameters():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1]])

    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(prompt, max_new_tokens=-1)

    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, temperature=0.0)

    with pytest.raises(ValueError, match="top_k"):
        model.generate(prompt, top_k=0)


def test_generation_with_top_k():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1, 2]])

    out = model.generate(prompt, max_new_tokens=5, top_k=2)
    assert out.shape == (1, 7)


def test_generation_stops_on_eos():
    config = TransformerConfig(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = TransformerLanguageModel(config)
    prompt = torch.tensor([[1]])

    # Make model deterministically output token 5 as highest probability next token
    def mock_forward(x):
        logits = torch.zeros(x.size(0), x.size(1), 20)
        logits[:, :, 5] = 100.0
        return logits, None

    model.forward = mock_forward

    out = model.generate(prompt, max_new_tokens=10, eos_id=5)
    # Token 5 is generated on first step, triggering early stop at length 2
    assert out.shape == (1, 2)
    assert out[0, 1].item() == 5

