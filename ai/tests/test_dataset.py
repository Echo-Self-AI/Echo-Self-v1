import pytest
import torch

from echoself_ai.dataset import NextCharacterDataset, load_corpus
from echoself_ai.tokenizer import CharacterTokenizer


def test_dataset_returns_shifted_sequences():
    text = "hello world"
    tokenizer = CharacterTokenizer(text)
    dataset = NextCharacterDataset(text, tokenizer, block_size=4)

    x, y = dataset[0]

    assert x.tolist() == tokenizer.encode("hell")
    assert y.tolist() == tokenizer.encode("ello")
    assert torch.equal(x[1:], y[:-1])


def test_dataset_length():
    text = "hello world"
    dataset = NextCharacterDataset(
        text,
        CharacterTokenizer(text),
        block_size=4,
    )

    assert len(dataset) == len(text) - 4


def test_invalid_block_size():
    with pytest.raises(ValueError, match="block_size"):
        NextCharacterDataset("hello", CharacterTokenizer("hello"), 0)


def test_text_must_be_longer_than_block_size():
    with pytest.raises(ValueError, match="more characters"):
        NextCharacterDataset("hello", CharacterTokenizer("hello"), 5)


def test_load_corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("hello world", encoding="utf-8")

    assert load_corpus(path) == "hello world"


def test_empty_corpus_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_corpus(path)
