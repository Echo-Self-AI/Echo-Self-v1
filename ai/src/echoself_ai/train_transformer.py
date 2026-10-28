import argparse
import math
from pathlib import Path
import random
from typing import Optional

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
    resume_from: Optional[Path] = None,
    tokenizer_type: str = "char",
    bpe_vocab_size: int = 50,
    use_cosine_scheduler: bool = False,
    warmup_steps: int = 0,
    min_lr: Optional[float] = None,
) -> dict:
    """Train a tiny decoder-only Transformer from scratch."""
    random.seed(seed)
    torch.manual_seed(seed)

    text = load_corpus(data_path)

    # 90% train / 10% validation text split strictly separated before tokenization
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    if tokenizer_type == "bpe":
        from echoself_ai.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train(train_text, target_vocab_size=bpe_vocab_size)
        train_raw_ids = tokenizer.encode(train_text)
        val_raw_ids = tokenizer.encode(val_text)
        vocab_chars = tokenizer.vocab
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = CharacterTokenizer(train_text)
        train_raw_ids = tokenizer.encode(train_text)
        val_raw_ids = [tokenizer.stoi[c] for c in val_text if c in tokenizer.stoi]
        vocab_chars = tokenizer.characters
        vocab_size = tokenizer.vocab_size

    train_ids = torch.tensor(train_raw_ids, dtype=torch.long)
    val_ids = torch.tensor(val_raw_ids, dtype=torch.long)


    model_config = TransformerConfig(
        vocab_size=vocab_size,
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
        resume_from=resume_from,
        use_cosine_scheduler=use_cosine_scheduler,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )

    print("=" * 60)
    print("EchoSelf AI — Decoder-Only Transformer Training")
    print("=" * 60)
    print(f"Corpus: {data_path} ({len(text)} characters)")
    print(f"Tokenizer: {tokenizer_type.upper()} (vocab size: {vocab_size})")
    print(f"Model parameters: {model.get_num_params():,}")
    print(
        f"Architecture: n_layer={n_layer}, n_head={n_head}, "
        f"n_embd={n_embd}, block_size={block_size}"
    )
    print(f"Training tokens: {len(train_ids)} | Validation tokens: {len(val_ids)}")
    print(f"Learning rate scheduler: {'Cosine with Warmup' if use_cosine_scheduler else 'Constant'}")
    print(f"Requested device: {device}")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_characters=vocab_chars,
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
        if tokenizer_type == "bpe":
            encoded_prompt = tokenizer.encode(prompt)
            if not encoded_prompt:
                encoded_prompt = [tokenizer.bos_id]
        else:
            valid_prompt_chars = [c for c in prompt if c in tokenizer.stoi]
            if not valid_prompt_chars:
                valid_prompt_chars = [tokenizer.characters[0]]
            encoded_prompt = [tokenizer.stoi[c] for c in valid_prompt_chars]

        prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=trainer.device)

        generated = eval_model.generate(
            prompt_tensor,
            max_new_tokens=100,
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
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer algorithm ('char' or 'bpe')",
    )
    parser.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=50,
        help="Target BPE vocabulary size when using --tokenizer-type bpe",
    )
    parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        help="Enable cosine annealing learning rate scheduler with warmup",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Linear warmup steps for scheduler",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning rate floor for scheduler",
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
        resume_from=args.resume_from,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        use_cosine_scheduler=args.use_cosine_scheduler,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
    )


if __name__ == "__main__":
    main()

