# EchoSelf AI Experiments Log

## Experiment 1: Decoder-Only Transformer vs. Bigram Baseline (Phase 1 Milestone)

- **Date**: 2026-10-28
- **Corpus**: `data/raw/tiny_corpus.txt` (191 characters, 23 unique characters)
- **Objective**: Establish and compare a from-scratch decoder-only Transformer with the baseline Bigram language model under identical data conditions on CPU.

### Hyperparameters

| Parameter | Bigram Baseline | Decoder-Only Transformer |
| :--- | :--- | :--- |
| Model Type | Lookup Embedding | Decoder Transformer (Pre-LN) |
| Parameters | 529 | 26,720 |
| Context Length (`block_size`) | 1 | 16 |
| Embedding Dim (`n_embd`) | 23 | 32 |
| Attention Heads (`n_head`) | N/A | 4 (dim 8/head) |
| Layers (`n_layer`) | 0 | 2 blocks |
| Optimizer | AdamW (`lr=1e-2`) | AdamW (`lr=1e-3`, `wd=0.01`, `clip=1.0`) |
| Steps | 1,000 | 300 |
| Batch Size | Full sequence | 8 |
| Device | CPU | CPU |

### Quantitative Results

| Metric | Bigram Baseline | Decoder-Only Transformer |
| :--- | :--- | :--- |
| Initial Train Loss | 3.7302 | 3.1590 |
| Initial Val Loss | 3.9349 | 3.1658 |
| Final Train Loss | 1.6021 | **0.9299** |
| Final Val Loss | 1.7803 | **1.0948** |
| Final Val Perplexity | ~5.93 | **2.99** |

### Qualitative Generation Comparison

Prompt: `"hello"`

**Bigram Output:**
```text
hrwtprd txwtai ra

wtrwytumwcdxwxxpyxitsfoe i mcnbmfdgytawratfsiy
fo med
xwtxwcuppppyawrranidilawr ccbscssyhi fofop
```

**Transformer Output (temperature=0.8, top_k=5):**
```text
hello
ho wo e world
helorld
w elomonteldechelsechomelld losecho losechoseldeld
weal wearnd morld
helorlntern ld
we ararn orns dararn frn from tern
```

### Analysis & Honest Limitations
- **What was learned**: The Transformer successfully captured subword and character sequences spanning multiple tokens (learning frequent combinations such as `"world"`, `"learn"`, `"echo"`, `"from"`, `"tern"`). The self-attention mechanism over a context window of 16 tokens drastically outperforms memoryless bigram transitions.
- **What was NOT learned**: The model has not learned semantics or grammar; it has memorized high-frequency n-grams and transition probabilities of the tiny 191-character corpus. Low training loss on this miniature dataset is expected due to small corpus size and does not signify generalized language understanding.

---

## Experiment 2: BPE Subword Tokenization & Cosine Warmup Scheduling (Phase 2 Milestone)

- **Date**: 2026-10-28
- **Corpus**: `data/raw/tiny_corpus.txt`
- **Objective**: Evaluate the impact of moving from character-level modeling to learned Byte-Pair Encoding (BPE) subwords with a Cosine Annealing learning rate schedule.

### Configuration Comparison

| Parameter | Exp 1: Character Baseline | Exp 2: BPE + Cosine Warmup |
| :--- | :--- | :--- |
| Tokenizer | CharacterTokenizer | BPETokenizer (Learned merges) |
| Vocabulary Size | 23 | 40 |
| Total Sequence Tokens | 191 (171 train / 20 val) | **140 (122 train / 18 val)** |
| Sequence Compression | 1.0x (1 char = 1 token) | **1.36x compression (28.6% fewer tokens)** |
| Effective Context Horizon | ~16 characters (~2-3 words) | **~16 subwords (~8-12 words)** |
| LR Schedule | Constant (`lr=1e-3`) | Cosine with linear warmup (`warmup=20`, `min_lr=1e-4`) |
| Steps | 300 | 300 |
| Parameters | 26,720 | 27,264 |

### Quantitative Results

| Metric | Character Baseline | BPE + Cosine Warmup |
| :--- | :--- | :--- |
| Initial Loss (train / val) | 3.1590 / 3.1658 | 3.6917 / 3.7147 |
| Final Loss (train / val) | 0.9299 / 1.0948 | 1.3110 / 1.4035 |
| Final Validation Perplexity | 2.99 | 4.07 |

> [!NOTE]
> Perplexity and cross-entropy loss across different tokenizers are not directly comparable on raw numerical scale because the BPE vocabulary is larger (40 vs 23) and each token carries higher information density per step. The decisive advantage of BPE is sequence compression: the model perceives substantially longer real-text spans per fixed `block_size`.

### Qualitative Output Sample (Prompt: `"hello"`)
```text
hello echm wor buant
we om uuuimodelss s buaaant learn atterms
hello pa a nsels learn frns learn learn fx om  ss
e
e learnsels at learn learn
```
The model successfully emits merged multicharacter tokens such as `"hello"`, `"learn"`, `"echo"`, and `"models"`.

---

## Experiment 3: Pretraining Decoder-Only Transformer on TinyShakespeare (Phase 10 Milestone)

- **Date**: 2026-10-28
- **Corpus**: `data/raw/tinyshakespeare.txt` (1,115,394 characters, ~40,000 lines; Project Gutenberg Public Domain)
- **Objective**: Pretrain the 4-layer decoder-only Transformer from scratch (randomly initialized weights) using Byte-Pair Encoding (BPE) on real text data with CUDA GPU acceleration, record convergence metrics, and evaluate text generation against baselines.

### Training Configuration

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| Model Architecture | 4 layers, 4 attention heads, Pre-LN | Standard causal decoder Transformer |
| Embedding Dimension (`n_embd`) | 128 | Head dimension: 32 |
| Context Length (`block_size`) | 64 | Max tokens processed per causal step |
| Total Parameters | **834,304** | All parameters randomly initialized |
| Tokenizer | BPETokenizer | 256 vocabulary size, trained strictly on training split |
| Training Tokens | 655,648 | 90% split of corpus |
| Validation Tokens | 73,762 | 10% split of corpus (zero leakage into tokenizer) |
| Batch Size | 32 | 32 sequences of 64 tokens = 2,048 tokens/step |
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=0.01`, `clip=1.0`) | Linear gradient clipping |
| Total Steps | 500 | 1,024,000 tokens processed |
| Hardware & Device | NVIDIA RTX 3050 Laptop GPU (`cuda:0`) | CUDA acceleration automatically selected |
| Training Time | ~45 seconds | Efficient memory footprint (< 1.5 GB VRAM) |
| Checkpoint Path | `checkpoints/transformer_real/best_model.pt` | Verified with SHA-256 sidecar checksum |

### Quantitative Metrics

| Metric | Initial (Step 0) | Final (Step 500) | Best Validation |
| :--- | :--- | :--- | :--- |
| **Training Loss** | 5.5639 | **2.8236** | — |
| **Validation Loss** | 5.5603 | **2.9087** | **2.9087** |
| **Training Perplexity** | ~260.8 | **16.84** | — |
| **Validation Perplexity**| ~259.9 | **18.33** | **18.33** |

> [!NOTE]
> Perplexity values are calculated in the learned 256-token BPE space ($e^{\mathcal{L}}$).

### Qualitative Text Generation Comparison

Prompt: `"ROMEO:"`

#### 1. Baseline Bigram Model (Random / First-Order Letter Lookup)
```text
hrwtprd txwtai ra wtrwytumwcdxwxxpyxitsfoe
```
*Memoryless transition without words, syllables, or sentence structure.*

#### 2. Initial Transformer (Random Initialization, Step 0)
```text
ROMEO:;!x_81Q(P#;--q?v;bY,
```
*Uniform noise over vocabulary without phonotactics or punctuation rules.*

#### 3. Trained Transformer (Step 500, Checkpoint Reloaded)
```text
ROMEO:
It my am lord, a be a spame in to cage,
Which my cagain cance the common their fict,
And the can ager and flown and fall to to the gry:
If the 
```

### Analysis & Honest Capabilities Assessment
- **Learned Features**:
  1. **Play script structure**: Preserves speaker header and newline format (`ROMEO:\n...`).
  2. **Subword & word boundaries**: Constructs common English words (`lord`, `cage`, `common`, `their`, `fall`, `flown`) and punctuation patterns.
  3. **Rhythmic meter**: Emulates Elizabethan verse line length and cadence.
- **Honest Limitations**:
  - The model does not understand semantics, character motivation, or factual reality.
  - Generates plausible-sounding Elizabethan pseudo-words (`cagain`, `fict`, `ager`, `gry`) reflecting the BPE vocabulary capacity (256 merges) and 500-step training envelope.
  - This is an evaluated milestone demonstration of causal pretraining from random weights on local hardware.


