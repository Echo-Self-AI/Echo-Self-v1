from pathlib import Path
import pytest
import torch

from echoself_ai.train_transformer import run_training


def test_run_training_pipeline_end_to_end(tmp_path: Path):
    # Create a small corpus
    corpus_file = tmp_path / "test_corpus.txt"
    corpus_file.write_text(
        "hello world\nhello echoself\nwe build intelligent systems\n"
        "language models learn from text\necho self learns patterns\n",
        encoding="utf-8",
    )

    ckpt_dir = tmp_path / "checkpoints"

    results = run_training(
        data_path=corpus_file,
        steps=20,
        batch_size=2,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        learning_rate=1e-2,
        seed=1337,
        device="cpu",
        prompt="hello",
        checkpoint_dir=ckpt_dir,
    )

    assert "initial_train_loss" in results
    assert "final_train_loss" in results
    assert "best_val_loss" in results
    assert "sample_generation" in results
    assert results["sample_generation"].startswith("hello")

    # Verify saved checkpoint files
    assert (ckpt_dir / "best_model.pt").exists()
    assert (ckpt_dir / "latest_model.pt").exists()
