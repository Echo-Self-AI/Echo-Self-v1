import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from echoself_ai.bigram import BigramLanguageModel
from echoself_ai.tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EchoSelf's bigram model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/tiny_corpus.txt"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    text = args.data.read_text(encoding="utf-8")
    if len(text) < 2:
        parser.error("Training corpus must contain at least two characters")

    tokenizer = CharacterTokenizer(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split sequentially: train on the first 90%, validate on the remainder.
    split = int(len(token_ids) * 0.9)
    split = max(1, min(split, len(token_ids) - 1))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def evaluate(ids: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(ids[:-1].unsqueeze(0))
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                ids[1:],
            )
        model.train()
        return loss.item()

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training characters: {len(train_ids)}")
    print(f"Validation characters: {len(val_ids)}")
    print(f"Initial train loss: {evaluate(train_ids):.4f}")
    print(f"Initial validation loss: {evaluate(val_ids):.4f}")

    model.train()
    for step in range(1, args.steps + 1):
        # Each character predicts the following character.
        logits = model(train_ids[:-1].unsqueeze(0))
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            train_ids[1:],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"Step {step:4d}/{args.steps} | "
                f"train loss: {loss.item():.4f} | "
                f"validation loss: {evaluate(val_ids):.4f}"
            )

    model.eval()
    prompt = "h"
    prompt_ids = torch.tensor(
        [[tokenizer.stoi[char] for char in prompt if char in tokenizer.stoi]],
        dtype=torch.long,
    )

    if prompt_ids.numel() == 0:
        prompt_ids = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(prompt_ids, max_new_tokens=200)[0].tolist()
    print("\nGenerated text:")
    print(tokenizer.decode(generated))

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "bigram.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_characters": tokenizer.characters,
            "vocab_size": tokenizer.vocab_size,
            "model_type": "bigram",
            "seed": args.seed,
            "steps": args.steps,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
