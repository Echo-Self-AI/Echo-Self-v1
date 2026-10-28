# EchoSelf AI — Dataset Provenance, Licensing & Preprocessing

This document records the exact sources, licenses, disk requirements, and preprocessing pipelines for datasets used in EchoSelf AI model pretraining.

---

## 1. Text Pretraining Dataset: TinyShakespeare

### Provenance & License

| Property | Value |
|---|---|
| **Dataset Name** | TinyShakespeare (Project Gutenberg Extract) |
| **Source URL** | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` |
| **Origin** | Public Domain works of William Shakespeare |
| **License** | Public Domain (Project Gutenberg license / MIT distribution) |
| **Redistribution** | Free, unrestricted commercial and research use |
| **Raw File Size** | 1.11 MB (1,115,394 characters) |
| **Token Space** | Trained BPE subword tokenizer ($\text{target\_vocab\_size} = 512$) |
| **Git Tracking** | Ignored by default in `.gitignore` (`data/raw/*`) |

### Evaluation: TinyShakespeare vs. TinyStories
- **TinyStories**: Contains synthetically generated children's stories by Microsoft Research (Ronan Eldan & Yuanzhi Li). The complete dataset archive is $>1.5\text{ GB}$ compressed ($>4\text{ GB}$ uncompressed), which is too large for rapid local pretraining under our local compute constraints without prior approval.
- **TinyShakespeare**: 1.11 MB raw text with rich dialogue, consistent syntax, and 40,000 lines of structured dialogue. It trains smoothly in minutes on our NVIDIA RTX 3050 (3.7 GB VRAM) while achieving genuine loss reduction and coherent vocabulary learning.

### Preprocessing & Splitting Pipeline
1. **Extraction & Deduplication**: Text is loaded as clean UTF-8 string with line-ending standardization (`\r\n` $\to$ `\n`).
2. **Train/Validation Separation**:
   - Sequential 90% / 10% split: First 90% of tokens for training, remaining 10% strictly reserved for validation.
   - Validation tokens are **never** seen during BPE vocabulary training or gradient optimization.
3. **BPE Tokenization**: Subword merges computed up to vocabulary size 512 on the training split.
4. **Context Chunking**: Sequence batches sampled with context window $\text{block\_size} = 64$.

---

## 2. Speech Pretraining Dataset: LJSpeech (Planned Phase 3)

| Property | Value |
|---|---|
| **Dataset Name** | The LJ Speech Dataset |
| **Source** | LibriVox recordings of non-fiction book readings |
| **License** | Public Domain (Creative Commons Zero / Dedication) |
| **Audio Format** | 16-bit PCM WAV, 22,050 Hz (standardized to 16,000 Hz for EchoSelf) |
| **Subsetting** | Initial training utilizes a curated 100-sample subset (~5-10 minutes of speech, ~15 MB) to verify gradient convergence within local VRAM limits before scaling. |
