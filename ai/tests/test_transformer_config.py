import pytest
import torch

from echoself_ai.transformer.config import TransformerConfig, get_device


def test_valid_default_config():
    config = TransformerConfig(vocab_size=32)
    assert config.vocab_size == 32
    assert config.block_size == 64
    assert config.n_embd == 64
    assert config.n_head == 4
    assert config.n_layer == 4
    assert config.head_dim == 16
    assert config.dropout == 0.0
    assert config.bias is True


def test_invalid_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        TransformerConfig(vocab_size=0)


def test_invalid_block_size():
    with pytest.raises(ValueError, match="block_size"):
        TransformerConfig(vocab_size=10, block_size=0)


def test_invalid_n_embd_or_head():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(vocab_size=10, n_embd=65, n_head=4)

    with pytest.raises(ValueError, match="n_head"):
        TransformerConfig(vocab_size=10, n_head=0)


def test_invalid_n_layer():
    with pytest.raises(ValueError, match="n_layer"):
        TransformerConfig(vocab_size=10, n_layer=0)


def test_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        TransformerConfig(vocab_size=10, dropout=-0.1)

    with pytest.raises(ValueError, match="dropout"):
        TransformerConfig(vocab_size=10, dropout=1.0)


def test_get_device():
    device = get_device(prefer_cuda=False)
    assert device.type == "cpu"

    device_auto = get_device(prefer_cuda=True)
    assert device_auto.type in ("cuda", "cpu")
