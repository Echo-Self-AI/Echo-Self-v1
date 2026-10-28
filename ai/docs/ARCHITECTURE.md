# EchoSelf AI Subsystem Architecture

## Overview

The `ai/` subsystem provides the machine learning, model training, inference, and personal AI capabilities for EchoSelf.

```text
ai/
├── configs/                  # Model and training hyperparameter configurations
├── data/
│   ├── raw/                  # Source text and media data (tiny smoke corpora)
│   └── processed/            # Processed token shards and features
├── checkpoints/              # Model weights and checkpoint states (.gitignore managed)
├── experiments/              # Experiment loss logs, configs, and metrics
├── docs/                     # Subsystem documentation, roadmap, model cards
├── src/echoself_ai/
│   ├── __init__.py
│   ├── tokenizer.py          # Character and subword tokenizers
│   ├── dataset.py            # Sequence data loaders and corpus reader
│   ├── bigram.py             # Baseline bigram language model
│   ├── train_bigram.py       # Bigram baseline training script
│   ├── transformer/          # From-scratch decoder-only Transformer components
│   ├── training/             # Trainer, learning rate schedules, checkpointing
│   ├── evaluation/           # Evaluation metrics (loss, perplexity, generation checks)
│   ├── voice/                # Audio processing, synthesis, voice cloning components
│   ├── vision/               # Avatar and computer vision pipeline
│   ├── video/                # Prompt-to-video generation pipeline
│   └── service/              # Lightweight AI serving API
└── tests/                    # pytest unit and integration test suite
```

## Language Model Foundation Design

Our Language Model foundation adheres to a clean decoder-only Transformer architecture (GPT-style) implemented strictly from PyTorch tensor operations:

1. **Token & Positional Embeddings**: Map discrete token IDs and absolute position indices to $d_{model}$ latent vectors.
2. **Causal Multi-Head Self-Attention**: Projects queries, keys, and values across $H$ heads with a lower-triangular causal mask:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   where $M_{i,j} = -\infty$ for $j > i$.
3. **Feed-Forward Network (FFN)**: Two-layer MLP with expansion factor (typically 4x) and GELU / ReLU activation.
4. **Pre-Layer Normalization & Residual Connections**: Stable gradient flow with LayerNorm applied before attention and FFN blocks.
5. **Output Projection**: Linear head mapping back to vocabulary logits for cross-entropy loss calculation and autoregressive token generation.
