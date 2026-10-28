from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from echoself_ai.tokenizer import CharacterTokenizer

class NextCharacterDataset(Dataset):
    """Create fixed-length input and next-character target sequences"""

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        block_size: int = 16,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be atleast 1")
        
        if len(text) <= block_size:
            raise ValueError(
                "Text must contain more characters than block_size"
            )
            
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.token_ids = torch.tensor(
            tokenizer.encode(text),
            dtype=torch.long,
        )
        
    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")
        
        x = self.token_ids[index: index + self.block_size]
        y = self.token_ids[index + 1: index + self.block_size + 1]
        
        return x, y
    
def load_corpus(path: str | Path) -> str:
    """Read a UTF-8 training corpus from disk."""
    text = Path(path).read_text(encoding="utf-8")
    
    if not text.strip():
        raise ValueError(f"Corpus is empty: {path}")
    
    return text