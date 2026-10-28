import torch
from torch import nn

class BigramLanguageModel(nn.Module):
    """Predict the next character using the current character."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")

        # Each character ID maps to a vector of next-character scores.
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, sequence_length, vocab_size)
        return self.token_embedding(inputs)

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        tokens = start_tokens

        for _ in range(max_new_tokens):
            logits = self(tokens[:, -1:])
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens