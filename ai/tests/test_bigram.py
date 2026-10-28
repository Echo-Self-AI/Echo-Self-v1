import pytest
import torch

from echoself_ai.bigram import BigramLanguageModel


def test_forward_shape():
    model = BigramLanguageModel(vocab_size=12)
    inputs = torch.randint(0, 12, (2, 5))

    logits = model(inputs)

    assert logits.shape == (2, 5, 12)


def test_model_has_trainable_parameters():
    model = BigramLanguageModel(vocab_size=8)

    assert sum(p.numel() for p in model.parameters()) == 64
    assert all(p.requires_grad for p in model.parameters())


def test_generate_adds_requested_tokens():
    model = BigramLanguageModel(vocab_size=8)
    start = torch.tensor([[1, 2]])

    result = model.generate(start, max_new_tokens=5)

    assert result.shape == (1, 7)
    assert torch.equal(result[:, :2], start)


def test_invalid_vocabulary_size():
    with pytest.raises(ValueError, match="vocab_size"):
        BigramLanguageModel(vocab_size=0)


def test_negative_generation_length():
    model = BigramLanguageModel(vocab_size=8)

    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(torch.tensor([[1]]), max_new_tokens=-1)