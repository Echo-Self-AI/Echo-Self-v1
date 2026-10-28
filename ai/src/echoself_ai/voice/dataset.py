from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from echoself_ai.voice.audio import load_wav, resample_waveform
from echoself_ai.voice.features import LogMelSpectrogram


class SpeechVocab:
    """Character-level vocabulary for phonetic speech acoustic modeling."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
    # Standard printable characters for English speech
    STANDARD_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,!?'-\""

    def __init__(self, characters: Optional[str] = None) -> None:
        chars = characters if characters is not None else self.STANDARD_CHARS
        self.c2i: dict[str, int] = {}
        self.i2c: dict[int, str] = {}

        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.c2i[tok] = idx
            self.i2c[idx] = tok

        offset = len(self.SPECIAL_TOKENS)
        for idx, ch in enumerate(chars):
            token_id = offset + idx
            self.c2i[ch] = token_id
            self.i2c[token_id] = ch

        self.pad_id = self.c2i[self.PAD]
        self.unk_id = self.c2i[self.UNK]
        self.bos_id = self.c2i[self.BOS]
        self.eos_id = self.c2i[self.EOS]

    @property
    def size(self) -> int:
        return len(self.c2i)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode lowercased text string into token IDs."""
        text = text.lower()
        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        for ch in text:
            tokens.append(self.c2i.get(ch, self.unk_id))

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to a text string."""
        chars: list[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_id, self.unk_id, self.bos_id, self.eos_id):
                continue
            chars.append(self.i2c.get(tid, ""))
        return "".join(chars)


@dataclass
class SpeechItem:
    """A single aligned speech training example."""

    clip_id: str
    text: str
    token_ids: torch.Tensor
    mel: torch.Tensor
    num_frames: int
    speaker_id: int = 0


class SpeechDataset(Dataset[SpeechItem]):
    """PyTorch Dataset loading aligned speech utterances and target log-mel spectrograms."""

    def __init__(
        self,
        metadata_path: str | Path,
        wavs_dir: str | Path,
        vocab: Optional[SpeechVocab] = None,
        target_sample_rate: int = 16000,
        n_mels: int = 80,
        split: str = "all",
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.wavs_dir = Path(wavs_dir)
        self.vocab = vocab or SpeechVocab()
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Speech metadata file not found: {self.metadata_path}")
        if not self.wavs_dir.is_dir():
            raise FileNotFoundError(f"Speech WAVs directory not found: {self.wavs_dir}")

        self.mel_extractor = LogMelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=512,
            hop_length=128,
            win_length=512,
            n_mels=n_mels,
        )

        # Parse metadata lines
        lines = self.metadata_path.read_text(encoding="utf-8").strip().splitlines()
        all_records: list[tuple[str, str]] = []

        for line in lines:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            clip_id = parts[0].strip()
            # If normalized transcript exists at index 2, prefer it; otherwise use raw text
            text = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else parts[1].strip()

            wav_path = self.wavs_dir / f"{clip_id}.wav"
            if wav_path.is_file() and wav_path.stat().st_size > 44:
                all_records.append((clip_id, text))

        if not all_records:
            raise ValueError(f"No valid audio records found in {self.metadata_path} and {self.wavs_dir}")

        if split == "all":
            self.records = all_records
        else:
            # Reproducible split
            import random
            rng = random.Random(seed)
            shuffled = list(all_records)
            rng.shuffle(shuffled)

            num_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0.0 else 0
            if split == "train":
                self.records = shuffled[num_val:]
            elif split == "val":
                self.records = shuffled[:num_val]
            else:
                self.records = shuffled

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SpeechItem:
        clip_id, text = self.records[idx]
        wav_path = self.wavs_dir / f"{clip_id}.wav"

        # Load audio and resample to target_sample_rate if needed
        waveform, sr = load_wav(wav_path)
        if sr != self.target_sample_rate:
            waveform = resample_waveform(waveform, orig_sr=sr, target_sr=self.target_sample_rate)

        # Extract 80-channel log-mel spectrogram: (1, n_mels, frames) -> (n_mels, frames)
        with torch.no_grad():
            mel = self.mel_extractor(waveform).squeeze(0)

        # Encode text to token IDs
        tokens = self.vocab.encode(text)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        return SpeechItem(
            clip_id=clip_id,
            text=text,
            token_ids=token_tensor,
            mel=mel,
            num_frames=mel.size(1),
            speaker_id=0,
        )


def collate_speech_batch(batch: list[SpeechItem]) -> dict[str, torch.Tensor | list[str]]:
    """Collate and pad a batch of variable-length SpeechItems for training."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_token_len = max(item.token_ids.size(0) for item in batch)
    max_mel_frames = max(item.mel.size(1) for item in batch)
    n_mels = batch[0].mel.size(0)

    batch_size = len(batch)
    # Pad tokens with 0 (PAD token)
    padded_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    # Pad log-mel with log(1e-5) floor (~ -11.51)
    mel_pad_val = math.log(1e-5)
    padded_mels = torch.full((batch_size, n_mels, max_mel_frames), fill_value=mel_pad_val, dtype=torch.float32)

    token_lens = torch.zeros(batch_size, dtype=torch.long)
    mel_lens = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    clip_ids: list[str] = []

    for i, item in enumerate(batch):
        t_len = item.token_ids.size(0)
        m_len = item.mel.size(1)

        padded_tokens[i, :t_len] = item.token_ids
        padded_mels[i, :, :m_len] = item.mel

        token_lens[i] = t_len
        mel_lens[i] = m_len
        speaker_ids[i] = item.speaker_id
        clip_ids.append(item.clip_id)

    return {
        "token_ids": padded_tokens,
        "target_mel": padded_mels,
        "speaker_id": speaker_ids,
        "token_lens": token_lens,
        "mel_lens": mel_lens,
        "clip_ids": clip_ids,
    }
