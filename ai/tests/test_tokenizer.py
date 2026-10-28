import pytest

from echoself_ai.tokenizer import CharacterTokenizer


def test_encode_and_decode_round_trip():
    tokenizer = CharacterTokenizer("hello world")

    text = "hello"
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded) == text


def test_vocabulary_is_deterministic():
    first = CharacterTokenizer("banana")
    second = CharacterTokenizer("banana")

    assert first.characters == second.characters
    assert first.encode("banana") == second.encode("banana")


def test_unknown_character_raises_error():
    tokenizer = CharacterTokenizer("hello")

    with pytest.raises(ValueError, match="Unknown characters"):
        tokenizer.encode("z")


def test_empty_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        CharacterTokenizer("")
