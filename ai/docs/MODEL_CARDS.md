# Model Card: EchoSelf Decoder-Only Transformer (v0.1)

## Model Overview
- **Model Architecture**: Autoregressive Decoder-Only Transformer (GPT-style) implemented from scratch with PyTorch primitives.
- **Layers**: 2 Pre-LayerNorm Transformer decoder blocks.
- **Attention**: Causal Multi-Head Self-Attention with 4 heads (8 dimensions per head).
- **Embedding**: Token embedding (`wte`) tied with output language model head (`lm_head`), plus learned absolute positional embedding (`wpe`).
- **Total Parameters**: 26,720 parameters.
- **Context Length**: 16 tokens (configurable up to hardware limits).

## Intended Use
- **Primary Use**: Educational and foundational research component for EchoSelf, verifying language modeling mechanics, gradient flow, causal masking, autoregressive generation, and local CPU training reproducibility.
- **Out of Scope**: Production conversational AI, open-ended generation, factual querying, or any domain requiring generalized reasoning.

## Training Data & Hardware
- **Corpus**: `data/raw/tiny_corpus.txt` (191 characters, synthetic smoke corpus).
- **Compute**: Local CPU (12 logical cores), execution time ~30s for 300 steps.
- **License**: Apache 2.0 / EchoSelf Project.

## Ethical & Safety Considerations
- **Privacy**: Trained solely on local synthetic data. No personal data, credentials, or private transcripts are included.
- **Limitations**: Tiny vocabulary and context length. Generations reflect strictly the local corpus statistical distribution.
