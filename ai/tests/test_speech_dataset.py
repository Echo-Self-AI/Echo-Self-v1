from pathlib import Path
import pytest
import torch

from echoself_ai.voice.dataset import (
    SpeechDataset,
    SpeechItem,
    SpeechVocab,
    collate_speech_batch,
)
from echoself_ai.voice.audio import save_wav


def test_speech_vocab_encode_decode():
    vocab = SpeechVocab()
    assert vocab.size > 40
    assert vocab.pad_id == 0

    text = "Hello, world!"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded)

    assert decoded == text.lower()

    # Unknown character mapping
    encoded_unk = vocab.encode("Hello \u03c0")  # Greek pi
    assert vocab.unk_id in encoded_unk


def test_speech_vocab_special_tokens():
    vocab = SpeechVocab()
    encoded = vocab.encode("test", add_special_tokens=True)
    assert encoded[0] == vocab.bos_id
    assert encoded[-1] == vocab.eos_id


def test_speech_dataset_loading_and_resampling(tmp_path: Path):
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    # Create dummy 22050 Hz 1-second audio files
    sample_rate = 22050
    t = torch.linspace(0, 1.0, sample_rate)
    waveform = 0.5 * torch.sin(2 * torch.pi * 440.0 * t)

    save_wav(wavs_dir / "LJ001-0001.wav", waveform, sample_rate)
    save_wav(wavs_dir / "LJ001-0002.wav", waveform, sample_rate)

    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(
        "LJ001-0001|Printing in the early days|Printing in the early days.\n"
        "LJ001-0002|Modern typography|Modern typography.\n",
        encoding="utf-8",
    )

    dataset = SpeechDataset(
        metadata_path=meta_file,
        wavs_dir=wavs_dir,
        target_sample_rate=16000,
        n_mels=80,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert isinstance(item, SpeechItem)
    assert item.clip_id == "LJ001-0001"
    assert item.text == "Printing in the early days."
    assert item.token_ids.dim() == 1
    assert item.token_ids.dtype == torch.long
    assert item.mel.shape[0] == 80  # 80 mel channels
    # 1 second at 16000 Hz with hop 128 = ~126 frames
    assert item.mel.shape[1] > 100
    assert not torch.isnan(item.mel).any()


def test_speech_dataset_train_val_split(tmp_path: Path):
    wavs_dir = tmp_path / "wavs"
    wavs_dir.mkdir()

    meta_lines: list[str] = []
    for i in range(10):
        clip_id = f"LJ001-{i:04d}"
        wav_path = wavs_dir / f"{clip_id}.wav"
        save_wav(wav_path, torch.zeros(16000), 16000)
        meta_lines.append(f"{clip_id}|Sentence {i}|Sentence {i}.\n")

    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text("".join(meta_lines), encoding="utf-8")

    train_ds = SpeechDataset(meta_file, wavs_dir, split="train", val_ratio=0.2, seed=42)
    val_ds = SpeechDataset(meta_file, wavs_dir, split="val", val_ratio=0.2, seed=42)

    assert len(train_ds) == 8
    assert len(val_ds) == 2

    train_clips = {train_ds[i].clip_id for i in range(len(train_ds))}
    val_clips = {val_ds[i].clip_id for i in range(len(val_ds))}

    # Train and val must be completely disjoint
    assert train_clips.isdisjoint(val_clips)


def test_collate_speech_batch():
    item1 = SpeechItem(
        clip_id="c1",
        text="short",
        token_ids=torch.tensor([1, 2, 3]),
        mel=torch.randn(80, 20),
        num_frames=20,
    )
    item2 = SpeechItem(
        clip_id="c2",
        text="longer text",
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        mel=torch.randn(80, 35),
        num_frames=35,
    )

    batch = collate_speech_batch([item1, item2])

    assert batch["token_ids"].shape == (2, 6)
    assert batch["target_mel"].shape == (2, 80, 35)
    assert (batch["token_lens"] == torch.tensor([3, 6])).all()
    assert (batch["mel_lens"] == torch.tensor([20, 35])).all()

    # Check padding values
    assert batch["token_ids"][0, 3].item() == 0  # Pad token
    assert batch["target_mel"][0, 0, 25].item() < -11.0  # Mel pad value
