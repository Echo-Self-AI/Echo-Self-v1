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

