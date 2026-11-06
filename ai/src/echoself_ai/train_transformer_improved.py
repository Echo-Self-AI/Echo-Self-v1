"""Improved Transformer training with experiment tracking and evaluation.

This module provides an enhanced training workflow that includes:
- Experiment tracking and configuration management
- Systematic evaluation beyond loss/perplexity
- Generation quality assessment
- Reproducible experimental results
"""

import argparse
import math
import random
from pathlib import Path
from typing import Optional, List

import torch

from echoself_ai.dataset import load_corpus
from echoself_ai.tokenizer import CharacterTokenizer
from echoself_ai.training.trainer import Trainer, TrainerConfig, load_checkpoint
from echoself_ai.transformer.config import TransformerConfig
from echoself_ai.transformer.model import TransformerLanguageModel
from echoself_ai.evaluation import (
    calculate_perplexity,
    generate_evaluation_samples,
    format_evaluation_report,
)
from echoself_ai.experiment import ExperimentTracker


DEFAULT_EVAL_PROMPTS = [
    "ROMEO",
    "The",
    "First Citizen",
    "What",
    "Now",
]


def run_improved_training(
    data_path: Path,
    steps: int = 500,
    batch_size: int = 16,
    block_size: int = 128,
    n_embd: int = 128,
    n_head: int = 4,
    n_layer: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "auto",
    checkpoint_dir: Path = Path("checkpoints/transformer_improved"),
    resume_from: Optional[Path] = None,
    tokenizer_type: str = "char",
    bpe_vocab_size: int = 256,
    use_cosine_scheduler: bool = True,
    warmup_steps: int = 50,
    min_lr: Optional[float] = None,
    eval_prompts: Optional[List[str]] = None,
    experiment_name: Optional[str] = None,
) -> dict:
    """Run improved Transformer training with evaluation.
    
    Args:
        data_path: Path to training corpus
        steps: Total training steps
        batch_size: Training batch size
        block_size: Context length
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_layer: Number of transformer layers
        dropout: Dropout probability
        learning_rate: Initial learning rate
        seed: Random seed for reproducibility
        device: Device ('cpu', 'cuda', or 'auto')
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint to resume from
        tokenizer_type: 'char' or 'bpe'
        bpe_vocab_size: Vocabulary size for BPE
        use_cosine_scheduler: Enable cosine annealing
        warmup_steps: Warmup steps for scheduler
        min_lr: Minimum learning rate
        eval_prompts: Custom evaluation prompts
        experiment_name: Name for experiment tracking
        
    Returns:
        Dictionary with training results and metrics
    """
    # Initialize experiment tracking
    tracker = ExperimentTracker()
    
    config_dict = {
        "data_path": str(data_path),
        "steps": steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device,
        "tokenizer_type": tokenizer_type,
        "bpe_vocab_size": bpe_vocab_size if tokenizer_type == "bpe" else None,
        "use_cosine_scheduler": use_cosine_scheduler,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
    }
    
    exp_name = experiment_name or f"transformer_{tokenizer_type}_s{steps}_d{n_embd}x{n_layer}"
    experiment_id = tracker.create_experiment(
        name=exp_name,
        experiment_type="language_model",
        config=config_dict,
        description=f"Transformer training with {tokenizer_type} tokenizer",
        tags=["transformer", tokenizer_type, "improved"],
    )
    
    print(f"Experiment ID: {experiment_id}")
    
    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and split data
    text = load_corpus(data_path)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Initialize tokenizer
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
    
    # Initialize model
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )
    model = TransformerLanguageModel(model_config)
    
    # Initialize trainer
    trainer_config = TrainerConfig(
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_interval=max(1, steps // 10),
        eval_batches=10,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        device=device,
        resume_from=resume_from,
        use_cosine_scheduler=use_cosine_scheduler,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )
    
    print("=" * 80)
    print("EchoSelf AI — Improved Transformer Training")
    print("=" * 80)
    print(f"Corpus: {data_path} ({len(text):,} characters)")
    print(f"Tokenizer: {tokenizer_type.upper()} (vocab size: {vocab_size})")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Architecture: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
    print(f"Context length: {block_size} tokens")
    print(f"Training tokens: {len(train_ids):,} | Validation tokens: {len(val_ids):,}")
    print(f"Batch size: {batch_size} | Training steps: {steps}")
    print(f"Learning rate: {learning_rate} | Scheduler: {'Cosine' if use_cosine_scheduler else 'Constant'}")
    print(f"Device: {device}")
    print("-" * 80)
    
    # Update experiment status
    tracker.update_experiment(experiment_id, status="running")
    
    # Train
    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_ids=train_ids,
        val_ids=val_ids,
        vocab_characters=vocab_chars,
    )
    
    training_results = trainer.train()
    
    # Calculate perplexities
    train_ppl = calculate_perplexity(training_results["final_train_loss"])
    val_ppl = calculate_perplexity(training_results["final_val_loss"])
    best_val_ppl = calculate_perplexity(training_results["best_val_loss"])
    
    print("-" * 80)
    print(f"Initial train loss: {training_results['initial_train_loss']:.4f}")
    print(f"Initial val loss:   {training_results['initial_val_loss']:.4f}")
    print(f"Final train loss:   {training_results['final_train_loss']:.4f} (ppl: {train_ppl:.2f})")
    print(f"Final val loss:     {training_results['final_val_loss']:.4f} (ppl: {val_ppl:.2f})")
    print(f"Best val loss:      {training_results['best_val_loss']:.4f} (ppl: {best_val_ppl:.2f})")
    print("=" * 80)
    
    # Evaluation: Load best checkpoint and generate samples
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if best_ckpt_path.exists():
        print("\nLoading best checkpoint for evaluation...")
        eval_model = TransformerLanguageModel(model_config).to(trainer.device)
        load_checkpoint(best_ckpt_path, device=trainer.device, model=eval_model)
        eval_model.eval()
        
        # Use custom or default prompts
        prompts = eval_prompts or DEFAULT_EVAL_PROMPTS
        
        print(f"Generating evaluation samples with {len(prompts)} prompts...")
        eval_results = generate_evaluation_samples(
            model=eval_model,
            device=trainer.device,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.8,
            top_k=5,
            samples_per_prompt=3,
        )
        
        # Format and display evaluation report
        eval_report = format_evaluation_report(eval_results)
        print("\n" + eval_report)
        
        # Save evaluation report
        eval_report_path = checkpoint_dir / "evaluation_report.txt"
        eval_report_path.write_text(eval_report)
        
        # Extract aggregate metrics
        all_repetition_scores = []
        all_unique_ratios = []
        for result in eval_results:
            for rep_metric in result['repetition_metrics']:
                all_repetition_scores.append(rep_metric['repetition_score'])
                all_unique_ratios.append(rep_metric['unique_ngrams_ratio'])
        
        avg_repetition = sum(all_repetition_scores) / len(all_repetition_scores) if all_repetition_scores else 0.0
        avg_unique_ratio = sum(all_unique_ratios) / len(all_unique_ratios) if all_unique_ratios else 0.0
        
        # Update experiment with comprehensive metrics
        tracker.update_experiment(
            experiment_id,
            status="completed",
            results=training_results,
            metrics={
                "final_train_loss": training_results["final_train_loss"],
                "final_val_loss": training_results["final_val_loss"],
                "best_val_loss": training_results["best_val_loss"],
                "train_perplexity": train_ppl,
                "val_perplexity": val_ppl,
                "best_val_perplexity": best_val_ppl,
                "avg_repetition_score": avg_repetition,
                "avg_unique_4gram_ratio": avg_unique_ratio,
            },
            artifacts={
                "best_checkpoint": str(best_ckpt_path),
                "latest_checkpoint": str(checkpoint_dir / "latest_model.pt"),
                "evaluation_report": str(eval_report_path),
            },
        )
        
        # Save experiment summary
        summary = tracker.format_experiment_summary(experiment_id)
        summary_path = checkpoint_dir / "experiment_summary.txt"
        summary_path.write_text(summary)
        
        print(f"\nExperiment summary saved to: {summary_path}")
        print(f"Evaluation report saved to: {eval_report_path}")
    
    else:
        tracker.update_experiment(
            experiment_id,
            status="completed",
            results=training_results,
            metrics={
                "final_train_loss": training_results["final_train_loss"],
                "final_val_loss": training_results["final_val_loss"],
                "best_val_loss": training_results["best_val_loss"],
                "train_perplexity": train_ppl,
                "val_perplexity": val_ppl,
                "best_val_perplexity": best_val_ppl,
            },
        )
    
    return {
        **training_results,
        "experiment_id": experiment_id,
        "train_perplexity": train_ppl,
        "val_perplexity": val_ppl,
        "best_val_perplexity": best_val_ppl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EchoSelf Transformer with improved evaluation"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/tiny_corpus.txt"),
        help="Path to training corpus",
    )
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Context length")
    parser.add_argument("--n-embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/transformer_improved"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer type",
    )
    parser.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=256,
        help="BPE vocabulary size",
    )
    parser.add_argument(
        "--use-cosine-scheduler",
        action="store_true",
        default=True,
        help="Use cosine annealing scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Warmup steps for scheduler",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name",
    )
    
    args = parser.parse_args()
    
    run_improved_training(
        data_path=args.data,
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        use_cosine_scheduler=args.use_cosine_scheduler,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
