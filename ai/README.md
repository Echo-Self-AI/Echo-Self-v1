# EchoSelf AI Layer

The AI subsystem for **EchoSelf**, an open, privacy-first personal AI platform.

## Subsystem Architecture & Philosophy

EchoSelf's AI layer is developed under strict engineering principles:
- **Free-first & local execution**: Zero paid cloud dependencies or external proprietary APIs required for core features. All initial experiments execute locally on standard hardware.
- **Built from scratch**: We implement model architectures (starting with our tiny decoder-only Transformer), loss functions, optimization loops, tokenization, and generation from PyTorch primitives rather than wrapping black-box pretrained LLMs.
- **Privacy & safety by design**: No private recordings, secrets, `.env` files, or unauthorized data are stored or uploaded. Checkpoints and raw datasets are kept local.
- **Microstep delivery**: Every functional change is implemented as an independently tested, verified, and committed microstep.

## Current Baseline (Phase 0 Audit)

- **Python Version**: 3.12.3
- **Environment & Package Manager**: `uv`
- **Core Dependencies**:
  - `torch` (2.14.1+cu130)
  - `numpy` (>=2.5.4)
  - `pytest` (>=9.2.0)
- **Hardware Environment**:
  - CPU: 12 logical cores
  - RAM: 7.45 GiB total (~2.67 GiB available)
  - GPU: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA capable, limited VRAM)
  - Execution profile: CPU-first default with optional CUDA device acceleration. Context lengths and batch sizes are kept lightweight.

### Implemented Components (Baseline)
- Character-level tokenizer: `src/echoself_ai/tokenizer.py`
- Sequence dataset & corpus loader: `src/echoself_ai/dataset.py`
- Bigram language model: `src/echoself_ai/bigram.py`
- Bigram training & generation script: `src/echoself_ai/train_bigram.py`
- Tiny training corpus: `data/raw/tiny_corpus.txt`
- Baseline checkpoint: `checkpoints/bigram.pt`
- Unit tests: 15 passing tests across `tests/test_tokenizer.py`, `tests/test_dataset.py`, `tests/test_bigram.py`.

## Quickstart

### 1. Environment Setup
Install dependencies using `uv`:
```bash
cd ai
uv sync
```

### 2. Run Tests
```bash
uv run pytest
```

### 3. Run Baseline Bigram Training
```bash
uv run python -m echoself_ai.train_bigram --steps 1000 --seed 42
```

### 4. Train from-scratch Decoder-Only Transformer
```bash
uv run python -m echoself_ai.train_transformer --steps 300 --device cpu --seed 42
```


## Documentation
- [ROADMAP.md](docs/ROADMAP.md) — Phased milestones, completed microsteps, and next steps.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Module hierarchy and interface designs.
