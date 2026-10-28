class CharacterTokenizer:
    """Convert text to character IDs and IDs back to text"""
    def __init__(self, text: str):
        if not text:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        
        self.characters = sorted(set(text))
        self.stoi = {
            character: index
            for index, character in enumerate(self.characters)
        }
        
        self.itos = {
            index: character
            for character, index in self.stoi.items()
        }
        
    @property
    def vocab_size(self) -> int:
        return len(self.characters)
    
    def encode(self, text: str) -> list[int]:
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(
                f"Unknown characters: {sorted(unknown)!r}"
            )
            
        return [self.stoi[character] for character in text]
    
    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.itos[index] for index in token_ids)
        except KeyError as exc:
            raise ValueError(
                f"Unknown token ID: {exc.args[0]}"
            ) from exc