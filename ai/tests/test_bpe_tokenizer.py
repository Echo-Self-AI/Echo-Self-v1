from pathlib import Path
import pytest

from echoself_ai.bpe_tokenizer import BPETokenizer


def test_bpe_training_and_merges():
    text = "low lower newest widest lowest"
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=30)

    assert tokenizer.vocab_size > len(tokenizer.special_tokens)
    assert len(tokenizer.merges) > 0


def test_bpe_encode_decode_round_trip():
    text = "hello world and welcome to echoself"
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=40)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_bpe_special_tokens():
    text = "machine learning"
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=30)

    encoded = tokenizer.encode(text, add_special_tokens=True)
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id

    # Decoding with skip_special_tokens=True strips them
    assert tokenizer.decode(encoded, skip_special_tokens=True) == text

    # Decoding with skip_special_tokens=False keeps them
    decoded_with_special = tokenizer.decode(encoded, skip_special_tokens=False)
    assert "<bos>" in decoded_with_special
    assert "<eos>" in decoded_with_special


def test_bpe_unknown_characters():
    tokenizer = BPETokenizer()
    tokenizer.train("cat and dog", target_vocab_size=25)

    # 'z' is not in the training text
    encoded = tokenizer.encode("zebra")
    assert tokenizer.unk_id in encoded


def test_bpe_save_and_load(tmp_path: Path):
    text = "reproducible subword tokenization for deep learning models"
    tokenizer = BPETokenizer()
    tokenizer.train(text, target_vocab_size=35)

    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(save_path)
    assert save_path.exists()

    loaded = BPETokenizer.load(save_path)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.merges == tokenizer.merges

    sample = "deep learning"
    assert loaded.encode(sample) == tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_bpe_invalid_arguments():
    tokenizer = BPETokenizer()

    with pytest.raises(ValueError, match="cannot be empty"):
        tokenizer.train("", target_vocab_size=20)

    with pytest.raises(ValueError, match="target_vocab_size"):
        tokenizer.train("abc", target_vocab_size=2)
