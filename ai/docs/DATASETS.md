# EchoSelf Dataset Registry & Provenance

This document tracks all training and evaluation corpora used across EchoSelf AI development, recording provenance, licenses, sizes, and privacy classifications.

## Policies & Compliance
1. **Zero Unauthorized Private Data**: Personal documents, private chat logs, passwords, or credentials must never be incorporated into training sets.
2. **Explicit Licensing**: Every corpus must have a confirmed permissive open license (e.g. Apache 2.0, MIT, CC-BY, Public Domain).
3. **Reproducible Hashing**: All dataset shards are verified against SHA-256 cryptographic checksums in `data/manifest.json`.

---

## Registered Datasets

### 1. `tiny_corpus` (Baseline Synthetic Smoke Corpus)
- **Path**: `data/raw/tiny_corpus.txt`
- **Manifest**: `data/manifest.json`
- **Source**: Local EchoSelf project synthetic benchmark sentences.
- **License**: Apache-2.0
- **Size**: 191 bytes (8 lines, 191 characters, 23 unique characters).
- **SHA-256 Checksum**: `486d8c361f08a1c1df27f751247823f291ad861fd03ac18ea450a6801bf60acc`
- **Intended Purpose**: Lightweight CPU unit and smoke testing for tokenizers, datasets, bigram baselines, and transformer architecture verification.
- **Privacy Classification**: Completely synthetic, non-sensitive.
