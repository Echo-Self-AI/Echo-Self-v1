from collections import Counter
import json
from pathlib import Path
from typing import Optional


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer implemented from scratch.

    Learns subword merge rules from raw text to compact sequences, allowing
    the language model to process longer conceptual contexts with the same
    sequence length budget.
    """

    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.merges: list[tuple[str, str]] = []
        self.vocab: list[str] = list(self.special_tokens)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def train(self, text: str, target_vocab_size: int = 100) -> None:
        """Train BPE merges on raw text until target_vocab_size is reached."""
        if not text:
            raise ValueError("Training text cannot be empty")
        if target_vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"target_vocab_size must be > {len(self.special_tokens)}, got {target_vocab_size}"
            )

        # Base vocabulary consists of unique individual characters
        unique_chars = sorted(set(text))
        self.vocab = list(self.special_tokens) + unique_chars
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.merges = []

        if len(self.vocab) >= target_vocab_size:
            return

        # Represent text words as list of characters
        # Split on whitespace but preserve words
        words = text.split()
        if not words:
            words = [text]

        # Each word represented as a tuple of tokens
        splits: list[list[str]] = [list(w) for w in words]

        num_merges = target_vocab_size - len(self.vocab)

        for _ in range(num_merges):
            # Count adjacent pair frequencies across all words
            pairs: Counter[tuple[str, str]] = Counter()
            for word_tokens in splits:
                for i in range(len(word_tokens) - 1):
                    pairs[(word_tokens[i], word_tokens[i + 1])] += 1

            if not pairs:
                break

            # Find most frequent pair
            best_pair, freq = pairs.most_common(1)[0]
            if freq < 1:
                break

            # Create new subword token
            new_token = "".join(best_pair)
            self.merges.append(best_pair)
            self.vocab.append(new_token)
            new_id = len(self.vocab) - 1
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # Apply merge to all words in corpus
            new_splits: list[list[str]] = []
            for word_tokens in splits:
                merged: list[str] = []
                i = 0
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == best_pair
                    ):
                        merged.append(new_token)
                        i += 2
                    else:
                        merged.append(word_tokens[i])
                        i += 1
                new_splits.append(merged)
            splits = new_splits

            if len(self.vocab) >= target_vocab_size:
                break

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token IDs using learned merge rules."""
        if not text:
            return []

        tokens: list[str] = []
        words = text.split(" ")
        for w_idx, word in enumerate(words):
            if w_idx > 0:
                # Retain space as a token or map it
                tokens.append(" ")
            word_tokens = list(word)
            for pair in self.merges:
                merged: list[str] = []
                i = 0
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == pair
                    ):
                        merged.append("".join(pair))
                        i += 2
                    else:
                        merged.append(word_tokens[i])
                        i += 1
                word_tokens = merged
            tokens.extend(word_tokens)

        # Map to IDs, fallback to unk_id for unknown tokens
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(self.bos_id)

        for token in tokens:
            token_ids.append(self.token_to_id.get(token, self.unk_id))

        if add_special_tokens:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to string."""
        chars: list[str] = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Serialize tokenizer vocabulary and merge rules to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Deserialize tokenizer from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.vocab = data["vocab"]
        tokenizer.token_to_id = {t: i for i, t in enumerate(tokenizer.vocab)}
        tokenizer.id_to_token = {i: t for i, t in enumerate(tokenizer.vocab)}
        return tokenizer
