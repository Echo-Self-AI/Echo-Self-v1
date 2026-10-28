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

## 2. Speech Pretraining Dataset: LJSpeech Subset (Phase 11)

### Provenance & License

| Property | Value |
|---|---|
| **Dataset Name** | The LJ Speech Dataset (Test Subset) |
| **Original Creators** | Keith Ito & Linda Johnson (2017) |
| **Origin & Speaker** | LibriVox recordings of non-fiction book readings by Linda Johnson |
| **Primary License** | **Public Domain (Creative Commons CC0 1.0 Universal)** |
| **Redistribution Terms** | Unrestricted public domain dedication — free for commercial, research, and educational use without royalty or copyright restrictions |
| **Curated Subset** | 32 verified clips (`LJ001-0001` through `LJ001-0032`) mirrored from standard public domain TTS verification suites |
| **Total Audio Duration** | ~2.5 minutes of continuous speech (~4,000 words across 32 utterances) |
| **Total Disk Size** | 9.33 MB (wavs) + 7 KB (metadata.csv) |
| **Audio Format** | 16-bit PCM WAV, mono, original sample rate 22,050 Hz |
| **EchoSelf Standardized Rate** | 16,000 Hz (resampled on ingestion using pure-PyTorch band-limited interpolation) |
| **Acoustic Features** | 80-channel log-mel spectrogram ($\text{n\_fft}=512$, $\text{hop\_length}=128$, $\text{win\_length}=512$, $16\text{ kHz}$) |
| **Git Tracking** | Ignored by default in `.gitignore` (`data/raw/*`) |

### Transcript Format & Vocabulary

Transcripts in `metadata.csv` are pipe-separated records:
```text
clip_id|raw_text|normalized_text
```
- Example: `LJ001-0001|Printing, in the only sense with which we are at present concerned...|Printing, in the only sense with which we are at present concerned...`
- Preprocessing tokenizes the normalized text characters into an ASCII character vocabulary (alphabet, punctuation, spaces, and padding/BOS tokens), or via BPE.

### Train/Validation Split
- **Training split**: 26 utterances (81.25%, ~7.6 MB)
- **Validation split**: 6 utterances (18.75%, ~1.7 MB, `LJ001-0027` to `LJ001-0032`)
- Validation clips are strictly held out to evaluate acoustic reconstruction error ($L_1$ mel loss) on unseen speech.

