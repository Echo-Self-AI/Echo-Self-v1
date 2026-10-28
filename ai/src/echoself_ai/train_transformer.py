import argparse
import math
from pathlib import Path
import random

import torch

from echoself_ai.dataset import load_corpus
from echoself_ai.tokenizer import CharacterTokenizer
from echoself_ai.training.trainer import Trainer, TrainerConfig, load_checkpoint
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel


def run_training(
    data_path: Path,
    steps: int = 300,
    batch_size: int = 8,
    block_size: int = 16,
    n_embd: int = 32,
    n_head: int = 4,
    n_layer: int = 2,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
    prompt: str = "hello",
    checkpoint_dir: Path = Path("checkpoints/transformer"),
) -> dict:
    """Train a tiny decoder-only Transformer from scratch."""
    random.seed(seed)
    torch.manual_seed(seed)

    text = load_corpus(data_path)
    tokenizer = CharacterTokenizer(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 90% train / 10% val split sequentially
    split = int(len(token_ids) * 0.9)
    split = max(block_size + 2, min(split, len(token_ids) - (block_size + 2)))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    model = TransformerLanguageModel(model_config)

    trainer_config = TrainerConfig(
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_interval=max(1, steps // 5),
        eval_batches=5,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    print("=" * 60)
    print("EchoSelf AI — Decoder-Only Transformer Training")
    print("=" * 60)
    print(f"Corpus: {data_path} ({len(text)} characters)")
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    print(f"Model parameters: {model.get_num_params():,}")
    print(
        f"Architecture: n_layer={n_layer}, n_head={n_head}, "
        f"n_embd={n_embd}, block_size={block_size}"
    )
    print(f"Training characters: {len(train_ids)} | Validation characters: {len(val_ids)}")
    print(f"Requested device: {device}")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_characters=tokenizer.characters,
    )

    results = trainer.train()

    train_ppl = math.exp(min(results["final_train_loss"], 20.0))
    val_ppl = math.exp(min(results["final_val_loss"], 20.0))

    print("-" * 60)
    print(f"Initial train loss: {results['initial_train_loss']:.4f}")
    print(f"Initial val loss:   {results['initial_val_loss']:.4f}")
    print(f"Final train loss:   {results['final_train_loss']:.4f} (ppl: {train_ppl:.2f})")
    print(f"Final val loss:     {results['final_val_loss']:.4f} (ppl: {val_ppl:.2f})")
    print(f"Best val loss:      {results['best_val_loss']:.4f}")
    print("=" * 60)

    # Verification: reload from saved best checkpoint and generate
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if best_ckpt_path.exists():
        eval_model = TransformerLanguageModel(model_config)
        load_checkpoint(best_ckpt_path, device=trainer.device, model=eval_model)
        eval_model.eval()

        # Encode prompt
        valid_prompt_chars = [c for c in prompt if c in tokenizer.stoi]
        if not valid_prompt_chars:
            valid_prompt_chars = [tokenizer.characters[0]]
        prompt_tensor = torch.tensor(
            [[tokenizer.stoi[c] for c in valid_prompt_chars]],
            dtype=torch.long,
            device=trainer.device,
        )

        generated = eval_model.generate(
            prompt_tensor,
            max_new_tokens=150,
            temperature=0.8,
            top_k=5,
        )
        generated_tokens = generated[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)

        print("\nGenerated text sample (temperature=0.8, top_k=5):")
        print("-" * 40)
        print(generated_text)
        print("-" * 40)
        results["sample_generation"] = generated_text

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EchoSelf's decoder-only Transformer model from scratch"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/tiny_corpus.txt"),
        help="Path to training text corpus",
    )
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--block-size", type=int, default=16, help="Context length")
    parser.add_argument("--n-embd", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layer", type=int, default=2, help="Transformer blocks")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--prompt", type=str, default="hello", help="Generation prompt")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/transformer"),
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    run_training(
        data_path=args.data,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        prompt=args.prompt,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
